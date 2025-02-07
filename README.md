# Bubble Finder

![License](https://img.shields.io/github/license/olivias004/bubble_finder)
![Python](https://img.shields.io/badge/Python-3.x-blue)

## Overview
Bubble Finder is a Python library designed to detect and analyze bubble-like structures in galaxies. This tool is intended for astrophysical research, particularly in studying feedback mechanisms, supernovae remnants, and other large-scale structures within galaxies.

## Features
- Automated identification of bubbles in astronomical images
- Parameter tuning for custom bubble detection
- Integration with astrophysical datasets (e.g., SDSS, HST, Legacy Surveys)
- Visualization tools for detected structures
- Customizable pipeline for astrophysical applications

## Installation
To install Bubble Finder, clone the repository and install the required dependencies:

```bash
git clone https://github.com/olivias004/bubble_finder.git
cd bubble_finder
pip install -r requirements.txt
```

## Usage
Here’s a basic example of how to use Bubble Finder to analyze a galaxy image:

```python
from bubble_finder import BubbleFinder

# Load an image
galaxy_image = "path/to/your/image.fits"

# Initialize the detector
bf = BubbleFinder(galaxy_image)

# Detect bubbles
bubbles = bf.detect_bubbles()

# Visualize results
bf.plot_bubbles()
```

## Dependencies
Ensure you have the following dependencies installed:
- `numpy`
- `scipy`
- `matplotlib`
- `astropy`
- `opencv-python`

You can install them using:
```bash
pip install numpy scipy matplotlib astropy opencv-python
```

## Contributing
Contributions are welcome! If you'd like to improve Bubble Finder, follow these steps:
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a pull request

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For any questions or collaboration inquiries, feel free to reach out via [GitHub Issues](https://github.com/olivias004/bubble_finder/issues).
