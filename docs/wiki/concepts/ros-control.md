# ROS Control

Sources: `dronet/repo/drone_control/dronet/README.md`, `dronet/repo/drone_control/dronet/dronet_perception/src/Dronet/Dronet.py`, `dronet/repo/drone_control/dronet/dronet_control/src/deep_navigation.cpp`.

## Perception Node

The ROS perception package loads the Keras model and weights, subscribes to camera images, preprocesses images, runs the network, and publishes `CNN_out` messages with:

- `steering_angle`
- `collision_prob`

The perception node has a `state_change` subscription that marks whether network output is being used, and a `land` callback that disables network output.

## Control Node

The C++ control node subscribes to `cnn_predictions` and `state_change`, then publishes `geometry_msgs/Twist` velocity commands when enabled.

The core forward-velocity formula is:

```text
desired_forward_velocity_m = (1.0 - probability_of_collision) * max_forward_index
```

The node then low-pass filters forward velocity and angular velocity:

- `alpha_velocity`, default `0.3`.
- `alpha_yaw`, default `0.5`.

If filtered forward velocity drops below the threshold implied by `critical_prob`, it sets forward velocity to zero. Default `critical_prob` is `0.7`.

## Operational Safety

The upstream README explicitly warns that DroNet directly produces flying commands and should be closely supervised. The same caution applies more strongly to any port or domain transfer.

