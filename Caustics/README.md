# Rock & Water Tracker

Real-time caustics demo, written in GLSL and running standalone in python with pygame and pygame_gui

## Dependencies

Requires python 3.12 or 3.13, numpy, pygame, pygame_gui, moderngl. I highly recommend creating a virtual environment in this directory and installing there, thats what I did! From there simply run venv\Scripts\activate.

## Running

From virtual environment, simply run the python app. Automatically set up to receive UDP packets from the touch tracker or computer vision tracker, easily adjusted to take values from the physical dial.

For best results, run the sim at .5 or so, upsampling enables better rendering at the target resolution. Art direction still WIP!