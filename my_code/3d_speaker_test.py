import argparse
import json
import os.path
import pickle
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchaudio
import uvicorn
import yaml
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from funasr import AutoModel

from speakerlab.utils.builder import build
from speakerlab.utils.config import build_config, Config

os.environ['LOCAL_RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # 允许任何源，或者指定特定源
    allow_credentials=True,  #
    allow_methods=['*'],  # 允许任何方法，或者指定方法
    allow_headers=['*'],  # 允许任何头，或者指定特定头
)


def remove_file(file_path_list):
    for filepath in file_path_list:
        if filepath.endswith('.wav') or filepath.endswith('.tmp') or filepath.endswith('.json') or filepath.endswith(
                '.pkl') or filepath.endswith('.rttm'):
            os.remove(filepath)
            print(f'[INFO]: removed file: {filepath}')
        else:
            print(f'[INFO]: file type: {filepath.rsplit(".", 1)[-1]} should not be removed')


def convert_to_wav(input_file: Path, output_file: Path):
    subprocess.run([
        'ffmpeg',
        '-i',  # 设定输入流
        input_file,
        '-ac',  # 指定音频声道数量
        '1',
        '-ar',  # 指定音频采样率
        '16000',
        '-sample_fmt',  # 采样位数
        's16',  # 有符号16位采样
        '-f',  # 强制输入或输出文件格式
        'wav',
        output_file,
    ],
        check=True,
    )


def vad_offline_not_realtime(wav_path):
    model = AutoModel(model=args.model_cache_dir + args.vad_model_id, model_version=args.vad_model_version)
    res = model.generate(input=wav_path)
    return res[0]


def save_vad_timestamp(timestamp, vad_output_file, wav_path):
    json_dict = {}
    for vad_t in timestamp['value']:
        start = round(vad_t[0] / 1000, 2)
        end = round(vad_t[1] / 1000, 2)
        subsegment_id = timestamp['key'] + '_' + str(start) + '_' + str(end)
        json_dict[subsegment_id] = {
            'file': wav_path,
            'start': start,
            'stop': end,
            'sample_rate': 16000
        }
    with open(vad_output_file, mode='w') as f:
        json.dump(json_dict, f, indent=2)
    print(f'[INFO]: VAD json is prepared in {vad_output_file}')


def prepare_subseg_json(vad_output_file, subseg_output_file):
    """将vad的结果分割成更小的子段"""
    with open(vad_output_file, 'r') as f:
        vad_json = json.load(f)
    subseg_json = {}
    print(f'[INFO]: Generate sub-segments...')
    for segid in vad_json:
        wav_id = segid.rsplit('_', maxsplit=2)[0]
        start = vad_json[segid]['start']
        end = vad_json[segid]['stop']
        subseg_start = start
        while subseg_start + args.subseg_dur < end:
            # 切割成[start + shift * i for i in ....]
            subseg_end = subseg_start + args.subseg_dur
            item = deepcopy(vad_json[segid])
            item.update({
                'start': round(subseg_start, 2),
                'end': round(subseg_end, 2)
            })
            subseg_id = f'{wav_id}_{round(subseg_start, 2)}_{round(subseg_end, 2)}'
            subseg_json[subseg_id] = item
            subseg_start += args.subseg_shift
        if subseg_start < end:
            subseg_start = min(end - args.subseg_dur, subseg_start)  # 最后一段小于1.5s的，end - subseg_start(右边) = 1.5
            subseg_start = max(subseg_start, start)  #
            item = deepcopy(vad_json[segid])
            item.update({
                'start': round(subseg_start, 2),
                'stop': round(end, 2)
            })
            subseg_id = f'{wav_id}_{round(subseg_start, 2)}_{round(end, 2)}'
            subseg_json[subseg_id] = item
    with open(subseg_output_file, mode='w') as f:
        json.dump(subseg_json, f, indent=2)
    print(f'[INFO]: Subsegments json is prepared in {subseg_output_file}')


def yaml_config_loader(conf_file_path):
    with open(conf_file_path, 'r') as f:
        conf_dict = yaml.load(f, Loader=yaml.FullLoader)  # 加载完整的yaml语言，避免任意代码执行
    return conf_dict


def extra_diar_embeddings( conf_file, subseg_json_path, embs_out_path, wav_path, gpus, use_gpu):
    conf = yaml_config_loader(conf_file)
    if 'campp' in args.extra_emb_model_id:
        obj = 'speakerlab.models.campplus.DTDNN.CAMPPlus'
        model_pt = 'campplus_cn_common.bin'
    else:
        obj = 'speakerlab.models.eres2net.ERes2Net_huge.ERes2Net'
        model_pt = 'pretrained_eres2net_aug.ckpt'
    model_config = {
        'revision': args.extra_emb_model_version,
        'model': {
            'obj': obj,
            'args': {'feat_dim': 80, 'embedding_size': 192}
        },
        'model_pt': model_pt,
    }
    feature_common = {
        'obj': 'speakerlab.process.processor.FBank',
        'args': {
            'n_mels': 80,
            'sample_rate': 16000,
            'mean_nor': True,
        },
    }
    pretrained_model = os.path.join(args.model_cache_dir + args.extra_emb_model_id, model_config['model_pt'])
    conf['embedding_model'] = model_config['model']
    conf['pretrained_model'] = pretrained_model
    conf['feature_extractor'] = feature_common

    with open(subseg_json_path, 'r') as f:
        subseg_json = json.load(f)
    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{gpus}')
        else:
            print("[WARNING]: Gpu is not available. Use cpu instead.")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    config = Config(conf)
    feature_extractor = build('feature_extractor', config)
    embedding_model = build('embedding_model', config)

    pretrained_state = torch.load(config.pretrained_model, map_location='cpu')
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.eval()
    embedding_model.to(device)

    embeddings = []
    wav, fs = torchaudio.load(wav_path)
    for segid in subseg_json:
        sample_start = int(subseg_json[segid]['start'] * fs)
        sample_stop = int(subseg_json[segid]['stop'] * fs)
        wav_seg = wav[:, sample_start:sample_stop]
        feat = feature_extractor(wav_seg).unsqueeze(0)
        feat = feat.to(device)
        with torch.no_grad():
            emb = embedding_model(feat).cpu().numpy()
        embeddings.append(emb)
    embeddings = np.concatenate(embeddings, axis=0)
    stat_obj = {
        'embeddings': embeddings,
        'times': [[subseg_json[i]['start'], subseg_json[i]['stop']] for i in subseg_json]
    }
    pickle.dump(stat_obj, open(embs_out_path, 'wb'))
    print(f'[INFO]: feature extract embedding pkl is prepared in {embs_out_path}')


def make_rttms(seg_list, out_rttm, wav_id):
    new_seg_list = []
    for idx, seg in enumerate(seg_list):
        seg_start, seg_end = float(seg[0][0]), float(seg[0][1])
        cluster_id = seg[1] + 1
        if idx == 0:
            new_seg_list.append([wav_id, seg_start, seg_end, cluster_id])
        elif cluster_id == new_seg_list[-1][3]:
            if seg_start > new_seg_list[-1][2]:
                new_seg_list.append([wav_id, seg_start, seg_end, cluster_id])
            else:  # 上一个的结束时间大于这次开始时间【有重叠】
                new_seg_list[-1][2] = seg_end
        else:
            if seg_start < new_seg_list[-1][2]:
                p = (new_seg_list[-1][2] + seg_start) / 2
                new_seg_list[-1][2] = p  # 将上一个人的结尾换成和下一个说话人的开始的平均时间
                seg_start = p
            new_seg_list.append([wav_id, seg_start, seg_end, cluster_id])
    with open(out_rttm, 'w+') as f:
        for seg in new_seg_list:
            f.write(f'SPEAKER {wav_id} 0 {seg[1]:.2f} {seg[2] - seg[1]}<NA> <NA> {seg[3]} <NA><NA>\n')
    print(f'[INFO]: Cluster rttm file is prepared in {out_rttm}')
    return new_seg_list


def cluster_and_postprocess(wav_path, embs_file, out_rttm_path):
    print('[INFO]: Start clustering')
    config = build_config(args.conf_file)
    cluster = build('cluster', config)
    wav_name = os.path.basename(wav_path)
    wav_id = wav_name.rsplit('.', 1)[0]
    with open(embs_file, 'rb') as f:
        stat_obj = pickle.load(f)
        embeddings = stat_obj['embeddings']
        times = stat_obj['times']
    labels = cluster(embeddings)
    new_labels = np.zeros(len(labels), dtype=int)
    uniq = np.unique(labels)  # 集合
    for i in range(len(uniq)):
        new_labels[labels == uniq[i]] = i
    seg_list = [(i, j) for i, j in zip(times, new_labels)]
    new_seg_list = make_rttms(seg_list, out_rttm_path, wav_id)
    return new_seg_list


@app.post('/sd_offline/')
async def sd_offline_wav(file: UploadFile = File(...)):
    print('save upload file')
    wav_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    audio_tmp = os.path.join(args.temp_dir, wav_id + '.tmp')

    with open(audio_tmp, 'wb') as buffer:
        await file.seek(0)
        chunk = await file.read(8192)
        while chunk:
            buffer.write(chunk)
            chunk = await file.read()
    wav_path = os.path.splitext(audio_tmp)[0] + '.wav'
    convert_to_wav(audio_tmp, wav_path)
    vad_timestamp = vad_offline_not_realtime(wav_path)

    vad_output_file = args.json_dir + '/vad.json'
    os.makedirs(os.path.dirname(vad_output_file), exist_ok=True)
    save_vad_timestamp(vad_timestamp, vad_output_file, wav_path)

    subseg_output_file = args.json_dir + '/subseg.json'
    os.makedirs(os.path.dirname(subseg_output_file), exist_ok=True)
    prepare_subseg_json(vad_output_file, subseg_output_file)

    os.makedirs(args.embs_dir, exist_ok=True)
    embs_out_path = os.path.join(args.embs_dir, wav_id + '.pkl')
    extra_diar_embeddings(
        conf_file=args.conf_file,
        subseg_json_path=subseg_output_file,
        embs_out_path=embs_out_path,
        wav_path=wav_path,
        gpus=args.gpus,
        use_gpu=args.use_gpu,
    )

    os.makedirs(args.rttm_dir, exist_ok=True)
    out_rttm_path = os.path.join(args.rttm_dir, wav_id + '.rttm')
    subseg_list = cluster_and_postprocess(wav_path, embs_file=embs_out_path, out_rttm_path=out_rttm_path)

    remove_file([audio_tmp, wav_path, vad_output_file, subseg_output_file, embs_out_path, out_rttm_path])
    print(subseg_list)

    # return subseg_list


if __name__ == '__main__':
    ROOT = r'D:/Speaker_main'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp',
        type=str,
        default=ROOT + '/my_code/exp',
        required=False
    )
    parser.add_argument(
        '--temp_dir',
        type=str,
        default=ROOT + '/my_code/my_data',
        required=False,
    )
    parser.add_argument(
        '--conf_file',
        type=str,
        default=ROOT + '/egs/3dspeaker/speaker-diarization/conf/diar.yaml',
        required=False,
    )
    parser.add_argument(
        '--gpus',
        type=str,
        default='0',
        required=False,
    )
    parser.add_argument(
        '--model_cache_dir',
        type=str,
        default=r'C:/Users/user/.cache/modelscope/hub',
        required=False,
        help='down model cache dir'
    )
    parser.add_argument(
        '--vad_model_id',
        type=str,
        default=r"/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        required=False,
        help='vad_model local path',
    )
    parser.add_argument(
        '--vad_model_version',
        type=str,
        default='v2.0.4',
        required=False,
    )
    parser.add_argument(
        '--extra_emb_model_id',
        type=str,
        default=r"/damo/speech_campplus_sv_zh-cn_16k-common",
        required=False,
        help='extract wav embedding model id',
    )
    parser.add_argument(
        '--extra_emb_model_version',
        type=str,
        default='v1.0.0',
        required=False,
    )
    parser.add_argument(
        '--subseg_dur',
        type=float,
        default=1.5,
        required=False,
        help='vad result to shorter dur',
    )
    parser.add_argument(
        '--subseg_shift',
        type=float,
        default=0.75,
        required=False,
        help='vad result shift len',
    )
    parser.add_argument(
        '--use_gpu',
        type=bool,
        default=True,
        required=False,
    )
    args = parser.parse_args()
    setattr(args, 'json_dir', f'{args.temp_dir}/json')
    setattr(args, 'embs_dir', f'{args.temp_dir}/embs')
    setattr(args, 'rttm_dir', f'{args.temp_dir}/rttm')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    uvicorn.run(app, host='127.0.0.1', port=8000)
