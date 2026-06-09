# DroNet Paper

Primary source: `dronet/RAL18_Loquercio.pdf` and extracted `dronet/paper_text.txt`.

## Core Claim

DroNet learns a drone navigation policy from ground-vehicle data rather than risky drone expert demonstrations in urban environments. It takes a single forward-looking camera image and emits:

- Steering angle, used to keep navigating and avoid obstacles.
- Collision probability, used to slow or stop when danger is likely.

## Training Idea

The paper combines two data sources:

- Steering from car-driving data, especially the Udacity self-driving-car dataset.
- Collision probability from a custom bicycle-collected outdoor dataset.

The conceptual transfer is that cars and bicycles are already integrated into urban environments, so they can provide large-scale visual navigation data without flying drones through unsafe demonstrations.

## Architecture

DroNet is described as a fast ResNet-8 with three residual blocks and two output heads. See [[model-architecture]].

## Control Idea

The paper maps steering output to yaw and uses collision probability to modulate forward velocity. The local results README states the yaw conversion as `steering * 90 degrees`, and the ROS control code uses `(1 - collision_probability) * max_forward_index` before filtering and thresholding. See [[ros-control]].

## Generalization Claim

The paper claims the policy learned from streets generalizes to drone flight and even some indoor environments such as corridors and parking lots. This should not be assumed to cover the local supermarket phone-video domain without validation.

