import numpy as np
from PIL import Image, ImageOps
import glob
import argparse

# slice an image into pieces of size 512

def readfile(f):
    im = Image.open(f);
    # width, height = im.size
    # im = im.resize((width//2, height//2))
    # print('after resize: ', im.size)
    # exit()
    im = np.array(im.convert('RGB'))
    return im

def readfile_scorpio(id,folder=''):
    ### folder = 'nksr/pdImages' and 'nksr/pdImagesMask'
    im = Image.open(folder+'/01'+str(id)+'.png');
    im = np.array(im.convert('RGB'))
    return im

def save_np_file(im,path):
    in2 = Image.fromarray(im.astype('uint8'))
    in2.save(path)

def pad_image_to_nearest_multiple(image, fill=0):

    height, width, _ = image.shape
    
    # Calculate the maximum side length required to be multiple of 512
    max_side_length = max(height, width)
    new_side_length = ((max_side_length - 1) // 512 + 1) * 512

    # Calculate the padding needed for each dimension
    pad_height = new_side_length - height
    pad_width = new_side_length - width

    # # Determine the shape of the input image
    # original_shape = image.shape
    
    # # Calculate the padding required on the first two dimensions
    # pad_height = (512 - (original_shape[0] % 512)) % 512
    # pad_width = (512 - (original_shape[1] % 512)) % 512
    
    # Initialize padding dimensions
    pad_dims = [(0, 0), (0, 0)]
    
    # If the image is 3D, add padding along the third dimension as well
    if len(image.shape) == 3:
        pad_dims.append((0, 0))
    
    # Update padding dimensions for the first two dimensions
    pad_dims[0] = (0, pad_height)
    pad_dims[1] = (0, pad_width)
    
    # Pad the image with zeros
    padded_image = np.pad(image, pad_dims, mode='constant')
    
    return padded_image    


parser = argparse.ArgumentParser(description="Pre-process data")
parser.add_argument('--data', help='NKSR_nuScenes_data | NKSR_KITTI | NKSR_MnCAV')
parser.add_argument('--delay', help='1.0 | 0.7 | 0.5')
parser.add_argument('--preprocess', action='store_true', help='preprocess | postprocess')

args = parser.parse_args()

data_folder = 'data/'+args.data+'/generateImages_'+args.delay+'s_delay/generateImagesNKSR_Image3'
mask_folder = 'data/'+args.data+'/generateImages_'+args.delay+'s_delay/generateImagesNKSR_mask'
files,maskfiles = glob.glob(data_folder+'/*.png'), glob.glob(mask_folder+'/*.png')
files.sort()
maskfiles.sort()

print(data_folder, '\n#files: ', len(files))
# exit()

for file,maskfile in zip(files[34:35],maskfiles[34:35]):
    fid = file[-8:-4]
    print(file[-8:])
    print('fid: ', fid)
    image = readfile(file)
    print(image.shape)
    mask = readfile(maskfile)
    # print('mask: ', mask.shape)
    # m2 = np.zeros_like(image)
    # m2[np.all(image < 10, axis=-1)] = 255
    # print('m2.shape: ', m2.shape)
    # exit()

    im = pad_image_to_nearest_multiple(image)
    mask = pad_image_to_nearest_multiple(mask)
    # m2 = pad_image_to_nearest_multiple(m2)
    print("resolution: ", im.shape)


    im = Image.fromarray(im.astype('uint8'))
    im.save('test_sets/Places/images/test'+str(fid)+'.png')

    mask = 255 - mask
    m = Image.fromarray(mask.astype('uint8'))
    m.save('test_sets/Places/masks/mask'+str(fid)+'.png')

    # m2 = 255 - m2
    # m2 = Image.fromarray(m2.astype('uint8'))
    # m2.save('test_sets/Places/masks/mask'+str(fid)+'_2.png')

    if args.postprocess:
        break
if args.preprocess
exit()


# files 
files = glob.glob('samples/*.png')
files.sort()

print('orig shape: ', image.shape)

file_save_path = 'samples/'+args.data+'_'+args.delay+'s_recons/inp'

for file in files:
    fid = file[-8:-4]
    
    inp = np.array(Image.open(file).convert('RGB'))
    save_np_file(inp[:image.shape[0],:image.shape[1]], file_save_path+str(fid)+'.png')
    print(fid, file_save_path+str(fid)+'.png')
    # exit()


