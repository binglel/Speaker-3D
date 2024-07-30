import argparse

import requests


def upload_with_request(args):
    url = f"http://{args.host}:{args.port}/{args.point_end}"
    with open(args.wav_path, 'rb') as f:
        response = requests.post(url, files={'file': f})  # request.post()会自动进行流式传输
    if response.status_code == 200:
        print('upload file success')
        return_res = response.content.decode('u8')
        print(return_res)
    else:
        print(f'upload file failed, status code {response.status_code}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--host',
        type=str,
        default="127.0.0.1",
        required=False,
    )
    parser.add_argument(
        '--port',
        type=str,
        default="8000",
        required=False,
    )
    parser.add_argument(
        '--point_end',
        type=str,
        default="sd_offline",
        required=False,
        choices=['sd_offline', ]
    )
    parser.add_argument(
        '--wav_path',
        type=str,
        default='./realtime_conversation.wav',
        required=False,
    )
    args = parser.parse_args()
    upload_with_request(args)
