from PIL import Image
import glob

# resize tif images in /path
for file in glob.glob('./path/*.TIF'):
    img = Image.open(file)
    resized_im = img.resize((256, 256), Image.ANTIALIAS)
    #jpg = img.convert('RGB')
    # print(file)
    resized_im.save('./path/' + file.split('\\')
                    [-1].split('.')[0] + '.png')
