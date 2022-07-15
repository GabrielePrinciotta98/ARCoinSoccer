# Coin SoccAR
Augmented Reality game inspired by the quite popular tabletop game [penny football](https://en.wikipedia.org/wiki/Penny_football).

# Description
We tried to Augment and Gamify the simple game of coin soccer.

The original game is played by flicking coins on a table.

Our game renders relevant objects on top of those objects to make it more
exciting in the Augmented World.

This project was developed with the collaborative efforts of:
Gabriele Princiotta, Levente Balazs Csik, Jakob Florian Goes, Leon Imhof, Towsif Zahin Khan

This project was made using:
OpenCV, FreeGlut, GLEW, GLFW

For a demo see: 
AR Coin SoccAR.mp4

https://github.com/GabrielePrinciotta98/ARCoinSoccer/blob/main/AR%20Coin%20SoccAR.mp4

# Playing Instructions:
- By default the game opens one of the test videos, to switch to the webcam, uncomment the CAMERA define in Application.cpp.
- Parameters like fov and coinSize for the tracking can be adjusted at runtime.
- Title screen and the victory screen can be skipped by pressing ENTER. Press F1 for toggling fullscreen.

# Setup instructions:
- Git: https://github.com/GabrielePrinciotta98/ARCoinSoccer
- To run the project, OpenCV (Tested with version 3.4.14), GLEW (2.1.0) and GLFW (3.3.7) must be linked as in the tutorials. 
- Additionally, we are using the freeglut library (Version 3.0.0, Binaries for windows available here: https://www.transmissionzero.co.uk/software/freeglut-devel/) 
  and the Windows Multimedia API (requires linking winmm.lib, should also work without this library when on a different OS, but there will be no sound effects.). 
- When linking GLEW and freeglut dynamically, there might be a runtime error if the dlls are not found on the path. 
  To fix this copy glew32.dll and freeglut.dll to the working directory. 
- The assets (PNG, OBJ, and WAV files) need to be in the working directory of the program. In visual studio this is the root directory of the project by default (i.e. just as in the git).

# Assets
- Soccerball model (https://www.turbosquid.com/de/3d-models/3d-max-soccer-ball/408588)
- Shoe model (https://www.turbosquid.com/3d-models/max-football-shoes/506661)
- Coin model (https://www.turbosquid.com/3d-models/coin-3d-model-1572874)

# Libraries
- OpenGL
- OpenCV
- GLEW
- GLFW
- GLUT
- lodepng (png decoding) https://github.com/lvandeve/lodepng
- Windows Multimedia API (optional for sound)
