# 3D Radio Environment Map Modeling for Aerial Networks

Graduate Project and Report, 3-credit, 1-semester work performed by Ryan Frost.
Timeframe: 01/2026 - 05/2026 (5 months)

## Background

I'm a Masters of Engineering in Computer Engineering candidate who earned a Bachelors of Science in General Engineering at JMU last May, 2025. I have spent the last year specializing in cyber-physical and embedded systems where physics and software meet. My motivation for working with Dr. Liu was to better understand the increasingly important role that Unmanned Aerial Vehicles (UAVs) are continuing to play in wireless systems. I saw this as a perfect opportunity to understand how the embedded system of a UAV integrates with physics-based radio communication.

## Problem Definition

### Starting Point

When I first started meeting with Dr. Liu in January, our discussions focused around NC State's unique Aerial Experimentation and Research Platform for Advanced Wireless (known as AERPAW). Shadab, one of Dr. Liu's students, was already investigating Radio Environment Map modeling, and he would be my primary contact for research questions and directions to explore. 

### Current State

UAVs are an incredibly useful tool for quickly deploying a wireless network over areas where traditional ground base stations are unavailable. Example scenarios include:

* Disaster Relief (where base stations have been destroyed)
* Remote areas
* Tactial operations

### Problem
Given this scenario, the natural next question to ask is how do we get the most out of a UAV-based wireless network? Answering this question requires an understanding of the 3D radio environemnt map to best position UAVs. You need to understand both where the UAVs will have a good connection, and how the UAVs can be orchestrated to provide the widest area of connectivity on the ground. This presents additional challenges from ground-based base stations and 2 dimensional ground connectivity:

* Changes in physics from 2D ground-based environment map to an aerial station that varies in altitude (3 dimensional map)
* Sparse Samples (the environment is now much larger in 3 dimensions and we'll be operating in areas where the environment has either changed significantly (disaster relief) or is entirely unknown/undocumented (remote areas and tactical operations)
* Limited power and compute on an arial station

## Project Layout

So we established scenarios that benefit from UAV-based wireless stations and the unique challenges that the solution presents. Next, we needed to plan a 5-month project to assist in solving the challenges. The project evolved as we better understood the problem space, and it can be summed up into four phases:

* Analysis on existing real-world datasets from published AERPAW datasets
* Dataset processing pipeline for suitability analysis and formatting
* Machine learning pipeline and baseline analysis (Line of Sight Path Loss and InceptionTime)
* Exploratory extreme learning machine analysis and comparison

## Analyizing AERPAW Datasets

### Problem
There are already existing digital twin-based simulated radio environment maps for tackling this exact challenge, so why did we want to look at AERPAW datasets? Simluated environments lack the nuances and fluctuations that exist in real-world scenarios, and those fluctuations act as crucial datapoints for a learning model to understand the environment. AERPAW provides a consistent experiment procedure and semi-reproducable results (environmental factors such as weather and ambiant signals still play a role). This means we have real-world datapoints from a real UAV connected to an actual base station.

When we started the project, we only had a basic understanding of AERPAW, but we did not know which datasets might be the most useful for our specific problem statement. My first task was to familiarize myself with the AERPAW procedure as well as summarize the available datasets for Shadab and Dr. Haihan to review. The results of this phase was the selection of Datasets 18, 22, and 24 for further analysis.

### Results
Presentation

## Dataset Processing

### Problem
Once we had our datasets, I was tasked with creating a processing pipeline to effectively combine the datasets into a standardized set that can be operated on as a group. Additionally, we wanted to further understand the characteristics of the datasets, such as the correlation between RSRP and datapoints in the same area as well as datapoints takent close together in time. At this point, we were mimicing some very recent work by folks at NC State (Gautham Reddy1, Kürşat Tekbıyık, Bryton Petersen, Antoine Lesage-Landry, Gunes Karabulut Kurt, and Ismail Güvenç). The result of this phase was better understanding of the datasets and a green light to proceed with machine learning analysis.

### Results
those two graphs

## Machine Learning Pipeline

### Problem
After the processing piepline was set up, I developed a PyTorch data loader to feed the data into machine learning models. We wanted to establish a baseline for where current researchers are at using the pipeline. First, we implemented the non-learning deterministic, physics-based Line of Sight Path Loss model. Then we implemented the deep-learning InceptionTime model, which is an established model for computer vision that can also be applied for modeling linear sequence data. 

### Results
similar to paper


## Extreme Learning Model

### Problem
Finally, we implemented the light-weight extreme learning machine model to understand how we might run a model on the UAV itself. 

This is where we diverged from what the paper itself proposed, which was a Transformer+GRU architecture. A ground-based deep learning model is another high-interest research topic and is useful as a network becomes more established with more samples and UAV-to-ground communication is reliable for model inference. However, our question was how do we quickly establish an understanding of the radio environment map when we might not have reliable access to heavy ground-based compute. 

This results of this phase was an understanding of how we might approach the challenges described:

* **2D to 3D Map:** A learning model can better map the unique physical characteristics of an environment as opposed to the idealized Line of Sight Path Loss model
* **Small Sample Size:** A light-weight, extreme learning machine can more quickly train on a small sample size.
* **Limited Power/Compute:** A deep-learning model requires a power-hungry GPU and long training times for back propagation. The extreme learning machine can retrain in a matter of seconds.

### Results
asdf


## Discussion

## Conclusion
