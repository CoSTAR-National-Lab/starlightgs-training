@echo off
REM Friendly launcher for the GS-IR one-command pipeline (Windows 10/11 + Docker Desktop + WSL2).
REM
REM   run.bat <scene_folder> <output_folder>
REM
REM Fills in the Docker flags so you don't have to remember them.
REM Override the image with:  set GSIR_IMAGE=...  before running.
setlocal

if "%~2"=="" (
  echo Usage: run.bat ^<scene_folder^> ^<output_folder^>
  echo.
  echo   ^<scene_folder^>   folder with your captured scene:
  echo                      NeRF-OSR : has a  train\pose\  folder
  echo                      COLMAP   : has a  sparse\      folder
  echo                      Blender  : has a  transforms_train.json  file
  echo   ^<output_folder^>  where the PLY, preview renders and checkpoints go
  exit /b 1
)

set "IMAGE=ghcr.io/costar/gs-ir:latest"
if not "%GSIR_IMAGE%"=="" set "IMAGE=%GSIR_IMAGE%"

if not exist "%~2" mkdir "%~2"

echo Scene : %~f1
echo Output: %~f2
echo Image : %IMAGE%
echo.

REM On Windows/Docker Desktop, files land as the current user already, so no --user needed.
REM %~n1 = the scene folder's name, so the output is named after YOUR folder (not /data).
docker run --rm --gpus all --shm-size=16g ^
  -e GSIR_SCENE=%~n1 ^
  -e GSIR_RESOLUTION -e GSIR_ITERS_S1 -e GSIR_ITERS_S2 ^
  -e GSIR_OCCLU_RES -e GSIR_OCCLUSION ^
  -e GSIR_TARGET_FRAMES -e GSIR_MATCHER -e GSIR_CAMERA ^
  -e GSIR_COLMAP_MAX_IMAGE_SIZE -e GSIR_COLMAP_GPU ^
  -v "%~f1":/data ^
  -v "%~f2":/output ^
  %IMAGE% /data

endlocal
