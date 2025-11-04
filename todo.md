# CLASH BOT TODO

## 1. Clear previous saved models on run to avoid unnecessary memory hogging

- When the code is run, all files in the `models` folder are deleted, other than the most recent model which will be loaded as a checkpoint.

## 2. Fix agent reward hacking in testing mode

- When using the agent in testing mode with `test.py`, it tends to reward hack by 'doing nothing' on every move. This should be addressed to prevent the agent from exploiting the reward system by avoiding actions.
