import os
import urllib.request
import zipfile


# Download COCO dataset
def download_coco_dataset(data_dir):
    """Download COCO keypoint dataset"""
    os.makedirs(data_dir, exist_ok=True)

    # URLs for COCO 2017 dataset
    urls = {
        'train_imgs': 'http://images.cocodataset.org/zips/train2017.zip',
        'val_imgs': 'http://images.cocodataset.org/zips/val2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    }

    for name, url in urls.items():
        filename = os.path.join(data_dir, url.split('/')[-1])

        if not os.path.exists(filename):
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, filename,
                                     reporthook=lambda b, bs, t: print(f'\r{b*bs/1e6:.1f}/{t/1e6:.1f} MB', end=''))
            print()

            # Extract zip file
            print(f"Extracting {name}...")
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print("Done!")