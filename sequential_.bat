@echo off

:: Activate the first virtual environment, evt. use Activate.ps1

CALL "C:\Users\Justus\Anaconda3\Scripts\activate.bat" suite2p

:: Run the first script
python C:\Users\Justus\calcium\data_suite2p.py

:: Deactivate the first environment
CALL conda deactivate

:: Activate the second virtual environment
CALL "C:\Users\Justus\Anaconda3\Scripts\activate.bat" cascade

:: Run the second script
python C:\Users\Justus\calcium\data_cascade.py

:: Deactivate the second environment
CALL conda deactivate

:: Activate the third virtual environment
CALL "C:\Users\Justus\Anaconda3\Scripts\activate.bat" data_env

:: Run the third script 
python C:\Users\Justus\calcium\data_data.py



:: keep terminal open 
pause