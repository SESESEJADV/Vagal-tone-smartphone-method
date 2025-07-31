Instructions to Run the Code

1. System Requirements
Operating System: Windows, macOS, or Linux.
Python Version: Python 3.6 or higher.

2. Required Python Libraries
The script depends on the following external libraries. Install them using pip (Python's package installer).
numpy
scipy
matplotlib
moviepy (for video processing)
pydub (for audio processing)
tkinter (usually included with Python, used for interactive dialogs)

3. How to Install the Required Libraries
Open a terminal or command prompt and run the following command:
pip install numpy scipy matplotlib moviepy pydub

4. How to Run the Script
Prepare Your Files:
	*Place the video file (recorded with the 	smartphone's flashlight and camera on the 	fingertip) and any necessary audio files in the 	same folder as the Python script.
Run the Script:
	*Open a terminal or command prompt.
	*Navigate to the directory containing the script.
	*Execute the script with the path to your video file as an argument. 
		For example: python analizador_pranayama.py "path/to/your/video.mp4"
Interactive Steps:
	*The script will first process the video to extract the photoplethysmography (spPPG) signal and 	detect heartbeats.
	*It will then process the audio to identify respiratory phases.
	*An interactive plot will appear, prompting you to enter two threshold values (in dB) for 	distinguishing breathing sounds from apnea and for differentiating inhalation from exhalation.
	*Enter the values in the pop-up windows and click OK.
Output:
	*The script will create a new folder (e.g., output/) and save the following files:
		*latidos.txt: A text file listing the timestamp of each detected heartbeat.
		*R-R.txt: A text file listing the R-R intervals (time between consecutive heartbeats).
		*ciclos.txt: A text file detailing the start and end times of each respiratory phase 			(Inhalation, Apnea, Exhalation).
		*grafico.png: A bar graph visualizing the detected respiratory cycles.
		*An Excel file (e.g., resultados.xlsx): A spreadsheet summarizing the average R-R 			interval for each respiratory phase.

5. Notes
Ensure that the video and audio recordings are synchronized, as they are assumed to start at the same time.
The quality of the spPPG signal can be affected by finger movement and ambient lighting. It is recommended to record in a stable, well-lit environment.
The audio recording should be made in a quiet environment to minimize background noise.
