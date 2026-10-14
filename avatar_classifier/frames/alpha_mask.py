from PIL import Image
from pathlib import Path

def merge(img_path, mask_path, output_path):
    mask = Image.open(mask_path).convert("L")
    # mask.show()
    img = Image.open(img_path).convert("RGBA")
    img.putalpha(mask)
    # img.show()
    img.save(output_path)

# 提取出来的原始帧素材，合成脚本
def main():
    merge(Path(__file__).parent / '1.png', Path(__file__).parent / '2.png', Path(__file__).parent / 'input_01.png')
    merge(Path(__file__).parent / '3.png', Path(__file__).parent / '4.png', Path(__file__).parent / 'input_02.png')
    merge(Path(__file__).parent / '5.png', Path(__file__).parent / '6.png', Path(__file__).parent / 'input_03.png')
    merge(Path(__file__).parent / '7.png', Path(__file__).parent / '8.png', Path(__file__).parent / 'input_04.png')
    merge(Path(__file__).parent / '9.png', Path(__file__).parent / '10.png', Path(__file__).parent / 'input_05.png')

if __name__ == "__main__":
    main()