# Monocular Depth Estimation and Point Cloud Generation

This repository contains the code for generating 3D point clouds from monocular images using the **Depth Anything** model. The goal of this project is to provide an easy-to-use setup for reproducing the depth estimation and point cloud generation process.

The process involves predicting depth from a single image, mapping the depth data onto the color image, and generating 3D point clouds.

## Description

This project utilizes **Depth Anything**, a monocular depth estimation model, to estimate the depth of a given image. The depth data is then combined with the color image to generate 3D point clouds, which can be visualized using **Open3D**. The main technologies used in this project are:

- **Depth Anything**: Monocular depth estimation model
- **Open3D**: For working with point clouds
- **PyTorch**: Deep learning framework used for the model

## Installation

To set up the project, you can use either **Conda** or **Pip** to install the required dependencies. Both installation methods are available in this repository.

### Option 1: Using Conda (Recommended)

1. Clone the repository:

    ```bash
    git clone https://github.com/my_name/Product.git
    cd Product
    ```

2. Create and activate the Conda environment using the `environment.yml` file:

    ```bash
    conda env create -f environment.yml
    conda activate <env_name>
    ```

3. Install the dependencies if you prefer using **Pip** (alternative):

    ```bash
    pip install -r requirements.txt
    ```

### Option 2: Using Pip

1. Clone the repository:

    ```bash
    git clone https://github.com/my_name/Product.git
    cd Product
    ```

2. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once the environment is set up, you can use the following command to run the program and generate the point cloud from an image:

```bash
python products.py
```

## License

Since this is a small project created as part of an internship, I am not using a formal license. However, you are welcome to use this code for personal, educational, or non-commercial purposes. Please be aware that this project is not intended for commercial use without proper modifications and review.

If you'd like to use this code for anything more than personal use, I would recommend reviewing the terms of the libraries and models used (such as **Depth Anything** and **DINOv2**) and ensuring you comply with their respective licenses.

---

## Contributing

This project was created as part of my internship, and it’s meant to be a small task that might help others. Contributions are not actively sought, but feel free to fork the repo or suggest improvements if you find the project useful. If you do decide to work with it, please make sure to adhere to any usage terms associated with the libraries or models used in this project.


## Citation

```bibtex
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}

@inproceedings{depth_anything_v1,
  title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle={CVPR},
  year={2024}
}

@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}

@misc{darcet2023vitneedreg,
  title={Vision Transformers Need Registers},
  author={Darcet, Timothée and Oquab, Maxime and Mairal, Julien and Bojanowski, Piotr},
  journal={arXiv:2309.16588},
  year={2023}
}
```

