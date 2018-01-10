# Generating_Music

Detailed writeup: "Final Writeup - Generating Music using an LSTM Network.pdf"

The Main.ipynb file is divided into 4 cells to individually run:

	1): import Python libraries
		INSTRUCTIONS:
			-run cell to import all the necessary Python libraries (REQUIRED)

	2): import .MIDI files
		INSTRUCTIONS:
			- run cell to read in the current working directory and specified MIDI files (REQUIRED)
		USER INPUT:
			- specify list of folders to extract music from
			- specify max # time steps required to retain each MIDI file
			- specify # pieces to set aside for independent validation during training

	3): Build model graph
		INSTRUCTIONS:
			- run cell to build the Input Kernel, timewise LSTM, notewise LSTM, loss, and optimizer graphs (REQUIRED)
		USER INPUT:
			- specify hidden sizes for timewise and notewise LSTM stages (a list for each)
			  num_t_units and num_n_units

	4): Train model
		DESCRIPTION:
			- runs a specified number of batches for training
			- every 100 batches it runs a validation batch and records validation loss in addition to training loss
			- every 500 batches it
		USER INPUT: 
			- specify # epochs (batches)
			- specify batch size (number of Note State Matrices)
			- specify # time steps to train on (must be less than max #time steps)
			- specify model name to restore (or None)
			- specify name to store model to
			- specify keep_prob = 1 - drop out rate
		
	5): Generate Averaged Test and Validation likelihoods
	
	6): Music Generation
		DESCRIPTION:
			- runs a for a specified amount of time steps, generating samples of the form Note_State_Batch, storing them, and feeding
				them back to the input of the next time step
			
		USER INPUT:
			- length of song in 16th note steps
			- batch size of generated music
			- specify keep_prob = 1 - drop out rate


All models, loss graphs, and generated midi files are automatically saved to a folder i.e. Generating_Music/Output/Save_Model_Name
		    where "Generating_Music" is the working directory and the subdirectory "Output" is reserved for all the generated files.
			models are restored from the same corresponding directories

Latest .MIDI files are under Ouptput/Piano_Midi2

		

	
