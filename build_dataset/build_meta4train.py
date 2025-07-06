import glob
import argparse
import json
import io
import os
import lmdb
from PIL import Image
from tqdm import tqdm
import cv2
import shutil


def save_lmdb(env_path, font_path_char_dict):
    """[saving lmdb]
    Args:
        env_path (string): folder root
        font_path_char_dict (list): img lists in folder
    Returns:
        [json]: {font name: [ch1, ch2, ch3, ch4, ....]}
    """
    env = lmdb.open(env_path, map_size=1024 ** 4)   # Create LMDB database with large enough space
    valid_dict = {}  # Store valid unicode list for each font

    for fname in tqdm(font_path_char_dict): 
        fontpath = font_path_char_dict[fname]["path"]   # Path to images of this font
        charlist = font_path_char_dict[fname]["charlist"]  # List of characters available for this font
        unilist = []  # Store corresponding unicode list

        for char in charlist:  
            img_path = os.path.join(fontpath, char + '.png')  # Try to locate .png image
            if not os.path.exists(img_path): 
                img_path = os.path.join(fontpath, char + '.jpg')  # If not exists, try .jpg
                print(img_path)

            if len(char) == 1:  # Only consider single character files  
                uni = hex(ord(char))[2:].upper()   # Get unicode (hex format, uppercase)
                unilist.append(uni)  # Add to unicode list
                char_img = cv2.imread(img_path, 0)  # Read image in grayscale

                # char_img = cv2.resize(char_img, (128, 128))  # Optional resizing (commented)

                char_img = Image.fromarray(char_img)  # Convert to PIL Image for encoding
                img = io.BytesIO()
                char_img.save(img, format="PNG")  # Save image to memory buffer in PNG format
                img = img.getvalue()

                lmdb_key = f"{fname}_{uni}".encode("utf-8")  # Generate LMDB key with font name and unicode

                with env.begin(write=True) as txn:
                    txn.put(lmdb_key, img)  # Store image in LMDB with corresponding key
            else:
                pass  # Skip filenames with length != 1 (unexpected cases)

        valid_dict[fname] = unilist  # Save valid unicode list for current font

    return valid_dict  # Return dict: {font_name: [unicode list]}


def getCharList(root):
    """[get all characters this font exists]

    Args:
        root (string): folder path

    Returns:
        [list]: char list
    """
    charlist = []  # Store available character list
    for img_path in (glob.glob(root + '/*.jpg') + glob.glob(root + '/*.png')):  
        ch = os.path.basename(img_path).split('.')[0]  # Extract filename without extension
        charlist.append(ch)  
    return charlist  # Return list of characters


def getMetaDict(font_path_list):
    """[generate a dict to save the relationship between font and its existing characters]
    Args:
        font_path_list (List): [training fonts list]

    Returns:
        [dict]: [description]
    """
    meta_dict = dict()  # Mapping from font name to its path and character list
    print("ttf_path_list:", len(font_path_list))
    for font_path in tqdm(font_path_list):  
        font_name = os.path.basename(font_path)  # Extract font name from path
        meta_dict[font_name] = {
            "path": font_path,
            "charlist": None
        } 
        meta_dict[font_name]["charlist"] = getCharList(font_path)  # Get available characters for this font
    return meta_dict  # Return meta information


def build_meta4train_lmdb(args):
    # saving directory
    out_dir = os.path.join(args.saving_dir, 'meta')  # Folder to save meta files
    lmdb_path = os.path.join(args.saving_dir, 'lmdb')  # LMDB output path
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(lmdb_path):
        shutil.rmtree(lmdb_path)  # Remove existing LMDB if any
    os.makedirs(lmdb_path, exist_ok=True)

    trainset_dict_path = os.path.join(out_dir, 'trainset_dict.json')
    # directory of your content_font
    content_font = args.content_font  # Path of content font

    # ===================================================================#
    train_font_dir = args.train_font_dir  # Path to training fonts
    validation_font_dir = args.val_font_dir  # Path to validation fonts

    dict_save_path = os.path.join(out_dir, "trainset_ori_meta.json")
    font_path_list = []

    font_chosen = []
    print(train_font_dir)  
    for font_name in os.listdir(train_font_dir):
        # print(font_name)
        font_chosen.append(os.path.join(train_font_dir, font_name))  # Append each font folder path

    font_chosen += glob.glob(validation_font_dir + "/*")  # Add validation fonts
    font_chosen = list(set(font_chosen))  # Remove duplicates

    print('num of fonts: ', len(font_chosen))  

    if content_font not in font_chosen:
        font_chosen.append(content_font)  # Ensure content font is included

    out_dict = getMetaDict(font_chosen)  # Build metadata for all fonts
    with open(dict_save_path, 'w') as fout:
        json.dump(out_dict, fout, indent=4, ensure_ascii=False)  # Save metadata to JSON

    valid_dict = save_lmdb(lmdb_path, out_dict)  # Store images to LMDB and get valid unicode list
    with open(trainset_dict_path, "w") as f:
        json.dump(valid_dict, f, indent=4, ensure_ascii=False)  # Save valid unicode list


def build_train_meta(args):
    train_meta_root = os.path.join(args.saving_dir, 'meta')
    # content
    content_font_name = os.path.basename(args.content_font)  # 'kaiti'
    # ==============================================================================#

    save_path = os.path.join(train_meta_root, "train.json")  # Output path for training meta
    meta_file = os.path.join(train_meta_root, "trainset_dict.json")  # Valid unicode list

    with open(meta_file, 'r') as f_in:
        original_meta = json.load(f_in)  # Load available unicode list
    with open(args.seen_unis_file) as f:
        seen_unis = json.load(f)  # Load seen characters
    with open(args.unseen_unis_file) as f:
        unseen_unis = json.load(f)  # Load unseen characters
    
    # all font names
    all_style_fonts = list(original_meta.keys())
    unseen_ttf_dir = args.val_font_dir  # Validation font directory
    unseen_ttf_list = [os.path.basename(x) for x in glob.glob(unseen_ttf_dir + '/*')]
    unseen_style_fonts = [ttf for ttf in unseen_ttf_list]  # Fonts for validation (unseen styles)

    # get font in training set
    train_style_fonts = list(set(all_style_fonts) - set(unseen_style_fonts))  # Exclude validation fonts

    train_dict = {
        "train": {},  # {font_name: [seen_unicodes]}
        "avail": {},  # {font_name: [all_unicodes]}
        "valid": {}   # Validation settings
    }

    for style_font in train_style_fonts:
        avail_unicodes = original_meta[style_font]  # Available unicode list
        train_unicodes = list(set.intersection(set(avail_unicodes), set(seen_unis)))  # Only keep seen ones
        train_dict["train"][style_font] = train_unicodes 

    for style_font in all_style_fonts:
        avail_unicodes = original_meta[style_font]
        train_dict["avail"][style_font] = avail_unicodes  # Save all available unicodes

    print("all_style_fonts:", len(all_style_fonts))
    print("train_style_fonts:", len(train_dict["train"]))
    print("val_style_fonts:", len(unseen_style_fonts))
    print("seen_unicodes: ", len(seen_unis))
    print("unseen_unicodes: ", len(unseen_unis))

    # validation set
    train_dict["valid"] = {
        "seen_fonts": list(train_dict["train"].keys()),  # Fonts in training set
        "unseen_fonts": unseen_style_fonts,  # Fonts used for validation
        "seen_unis": seen_unis,  # Seen character set
        "unseen_unis": unseen_unis,  # Unseen character set
    }

    with open(save_path, 'w') as fout:
        json.dump(train_dict, fout, ensure_ascii=False, indent=4)  # Save final training meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saving_dir", help="directory where your lmdb file will be saved")
    parser.add_argument("--content_font", help="root path of the content font images")
    parser.add_argument("--train_font_dir", help="root path of the training font images")
    parser.add_argument("--val_font_dir", help="root path of the validation font images")
    parser.add_argument("--seen_unis_file", help="json file of seen characters")
    parser.add_argument("--unseen_unis_file", help="json file of unseen characters")
    args = parser.parse_args()

    build_meta4train_lmdb(args)  # Step 1: generate LMDB and basic metadata
    build_train_meta(args)  # Step 2: prepare training/validation splits



