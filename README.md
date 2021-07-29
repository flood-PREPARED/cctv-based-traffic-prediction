# CCTV based traffic prediction
Prediction of traffic volume with Random Forest machine learning using CCTV cameras

## Usage
There are a number of steps required to run a prediction, with a model required for generating a prediction for a single camera location

### training


### predictions
Done with the predict.py file.

Run with the run() method. Accepts a camera name, but if not provided will search the models directory and run for each model present. If a camera name is passed, will check the model exists and run for the single model. Outputs are stored in the outputs directory.
* from python: 
  * `predict.run()`
* running in docker:
  * docker script calls predict script and runs for all models present
  * `docker build . -t cctv`
  * run for all existing models saving outputs to a local outputs directory
  * `docker run -v <local absolute path>outputs:/outputs -t cctv`
