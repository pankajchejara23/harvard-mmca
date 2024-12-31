# **Multimodal Collaboration Analytics**


This repository contains visualization codes developed in collaboration with Harvard University's Learning, Innovation, and Technology (LIT) Lab. The project focuses on Multimodal Collaboration Analytics (MMCA), utilizing various sensor technologies to gain insights into group behaviors during collaborative activities, aiming to enhance group collaboration.

## Overview

Multimodal Collaboration Analytics (MMCA) is an emerging field that leverages data from multiple sources—such as audio, video, physiological sensors, and digital interactions—to analyze and improve collaborative processes. In the past two decades, there has been an increasing number of research studies investigating collaboration using multimodal data. 

A synthesis of all those studies is crucial for new researchers as well as experienced ones to have updated understanding of state-of-the-art.  This repository contains codes aim to provide 

- **MMCA Review Library**: A Python library providing access to an extensive literature review dataset on MMCA studies published since 2000. It offers utility functions for filtering papers by time intervals and plotting trends for various coded attributes (e.g., metrics, outcomes).

- **Visualization Tools**: Code and resources for visualizing data related to MMCA, aiding in the analysis and interpretation of multimodal datasets.

## Repository Structure

- `dataset/`: Contains datasets used for analysis and visualization.

- `figures/`: Includes generated figures and visualizations.

- `source_codes/`: Cotains the source code for data processing and visualization tools.

- `src/`: Directory for render hosting. Live at: [https://harvard-mmca.onrender.com/](https://harvard-mmca.onrender.com/)

- `app.py`: Main application file for deployment on render.

- `requirements.txt`: List of Python dependencies required to run the code.

- `Dashboard.md`: Documentation for the interactive dashboard included in the project.

- `MMCA_library.md`: Documentation for the MMCA Review Library, including usage instructions.

## Getting Started

### Prerequisites

Ensure you have Python installed (version 3.6 or higher). Install the required dependencies using:

```bash
pip install -r requirements.txt
```

### Usage

* MMCA Review Library: Refer to [MMCA_library.md](https://github.com/pankajchejara23/harvard-mmca/blob/main/MMCA_library.md) for detailed instructions on accessing and utilizing the literature review dataset.

* Visualization Dashboard: Execute the following command after installing the required pacakges.

```bash
cd source_codes
python3 dashboard_mmca_cscw_v6.py
```
![](figures/visualizer.gif)

### License

This project is licensed under the MIT License. 

### Acknowledgments

This work was conducted in collaboration with **Harvard University's Learning, Innovation, and Technology (LIT) Lab**. Special thanks to [**Prof. Bertrand Schneider**](https://www.gse.harvard.edu/directory/faculty/bertrand-schneider) for his guidance and all researchers in the **MMCA community**.