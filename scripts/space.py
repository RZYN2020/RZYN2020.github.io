import argparse
import pangu

parser = argparse.ArgumentParser(description='中英文空格格式化工具')
parser.add_argument('input', help='输入文件路径')
parser.add_argument('--output', help='输出文件路径')
args = parser.parse_args()

with open(args.input, 'r', encoding='utf-8') as f:
    content = f.read()
processed = pangu.spacing(content)

if args.output:
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(processed)
else:
    print(processed)