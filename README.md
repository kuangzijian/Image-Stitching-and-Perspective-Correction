<!-- PROJECT LOGO
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Best-README-Template</h3>

  <p align="center">
    An awesome README template to jumpstart your projects!
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
  </p>
</p>
-->


<!-- TABLE OF CONTENTS 
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)
-->


<!-- ABOUT THE PROJECT -->
## Image-Stitching-and-Perspective-Correction
Image Stitching and Perspective Correction

This project focuses on topics:
* Implement DLT algorithm and the normalized DLT version to perform perspective correction to license plates.
* Perform Image warping and stitching using OpenCV for the warping and visualise.
* Use OpenCV to detect and visualise the lines in the image and Compute the Vanishing point and visualise on the image.

### Built With
* [Pytorch](https://github.com/pytorch)

### Prerequisites
```sh
1. Clone the repo
2. pip install -r requirements.txt
```

### Implement DLT algorithm and the normalized DLT version to perform perspective correction to license plates.

```
- perspective correction to license plates using DLT algorithm
python dlt\eval.py -i dlt/images -n dlt/gt

- perspective correction to license plates using normalized DLT algorithm
python dlt\eval.py -i dlt/images -n dlt/gt --norm

The visualized results will show up once program starts running:
```
![1](https://github.com/kuangzijian/Group-Activity-Recognition/blob/master/read_me_pictures/experiments.png)

### Perform Image warping and stitching using OpenCV for the warping and visualise.

```
- image stitching uses the ORB feature detection to merge "Picture1.jpg" and "Picture2.jpg"
python ImageStitching_ORB.py

- image stitching uses the SFIT feature detection to merge "knee1.jpg" and "knee2.jpg"
python ImageStitching_SIFT.py

```

### Use OpenCV to detect and visualise the lines in the image and Compute the Vanishing point and visualise on the image.

```
- image stitching uses the ORB feature detection to merge "Picture1.jpg" and "Picture2.jpg"
python ImageStitching_ORB.py

- image stitching uses the SFIT feature detection to merge "knee1.jpg" and "knee2.jpg"
python ImageStitching_SIFT.py

```


## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

## References
NCC https://github.com/rogerberm/pytorch-ncc

SSIM (https://github.com/Po-Hsun-Su/pytorch-ssim)

