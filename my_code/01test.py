import argparse
import os.path
import time

from matplotlib import pyplot as plt
from modelscope.pipelines import pipeline


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='damo/speech_eres2net-large_speaker-diarization_common',
        required=False,
        help='speaker-diarization model',
    )
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0.0',
        required=False,
        help='speaker-diarization model version',
    )
    parser.add_argument(
        '--path',
        type=str,
        default='my_code/realtime_conversation.wav',
        required=False,
        help='test_wav_path, file or dir'
    )
    parser.add_argument(
        '--root',
        type=str,
        default='D:/Speaker_main/',
        required=False,
        help='Used when the relative path is wrong',
    )
    args = parser.parse_args()
    return args


def speaker_diarization_test(args, wav_path):
    sd_pipeline = pipeline(
        task='speaker-diarization',
        model=args.model,
        model_revision=args.version,
    )
    print('Start inference')
    start = time.time()
    result = sd_pipeline(wav_path)
    # 如果有先验信息，输入实际的说话人数，会得到更准确的预测结果
    # result = sd_pipeline(input_wav, oracle_num=2)

    print(f"Inference takes time: {time.time() - start:2f}s")
    return result['text']


def plot_gantt_chart(data, annotation=''):
    # 绘制甘特图
    spks = len(set([row[2] for row in data]))
    colors = ['r', 'g', 'b', 'c', 'y', 'black', 'm', 'yellow', 'pink', 'tan', 'orange', 'gold', 'cyan']

    plt.figure(figsize=(12, 6), dpi=600)
    for i, (start, end, speaker) in enumerate(data):
        plt.plot(
            [start, end],
            [speaker, speaker],
            color=colors[speaker % len(colors)],
            linewidth=6,
        )

    plt.xlabel('Time(S)')
    plt.ylabel('Speaker')
    plt.yticks([i for i in range(spks)], [f'Speaker {i}' for i in range(spks)])
    plt.title(f'Speaking Time Segments of Each Speaker {annotation}')
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(10))
    plt.grid(which='both', linestyle='--', linewidth=0.5)

    plt.show()


def main():
    args = get_args()
    print(args)
    if not os.path.exists(args.root + args.path):
        raise f'path: {args.path} not Found Error'
    if os.path.isfile(args.root + args.path):
        time_segments = speaker_diarization_test(args, args.root + args.path)
        plot_gantt_chart(time_segments)
    elif os.path.isdir(args.root + args.path):
        for filename in os.listdir(args.root + args.path):
            if filename.endswith('.wav'):
                filepath = os.path.join(args.root, args.path, filename)
                time_segments = speaker_diarization_test(args, filepath)
                plot_gantt_chart(time_segments, filename)


if __name__ == '__main__':
    main()
