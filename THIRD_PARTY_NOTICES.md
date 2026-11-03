# Third-party components

Sightline's original code and documentation are maintained by Yousif Lubbad.
The repository license does not replace any dependency's license or claim
ownership of its code, datasets, or model weights.

| Component | Purpose | Upstream |
| --- | --- | --- |
| NumPy | Pixel arrays | https://numpy.org/ |
| Pillow | Load RGB images | https://python-pillow.org/ |
| MSS | Optional desktop capture | https://github.com/BoboTiG/python-mss |
| PyAutoGUI | Optional mouse input | https://github.com/asweigart/pyautogui |
| EasyOCR | Optional text recognition | https://github.com/JaidedAI/EasyOCR |
| PyTorch and torchvision | EasyOCR model execution | https://pytorch.org/ |
| OpenCV and other transitive packages | EasyOCR image processing | https://github.com/JaidedAI/EasyOCR/blob/v1.7.2/requirements.txt |

Dependencies are installed separately; their source and license files are not
replaced by this project. EasyOCR downloads model weights from its upstream
distribution when explicitly enabled. No model weights or downloaded screenshots
are committed here. Model and dataset rights
must be checked separately before redistributing them.

The demo app and test data generators are written for Sightline. They use local
system fonts, which remain under their own licenses. Test tools and GitHub Actions
also retain their upstream terms. This notice is not a claim of complete legal
clearance for every possible downstream distribution.
