# Data Driven Pitch Selection
#### Joey Asperger and Jorge Garcia

### FILE DESCRIPTIONS 

* grouped_batters.py: This runs all the our machine learning algorithms, including
the k-means to cluster the groups of hitters and stochastic gradient descent. This
contains all our feature extractors and functions to train and test the algorithms.
It also interacts directly with the mongoDB instance containing all the data.
When run with `python grouped_batters.py`, this will run k-means and train the regressor 
with stochastic gradient descent on 200,000 training pitches for 100 iterations. It will
then print the training error and then the testing error on a separate test set of 100,000 
pitches. This also saves the trained regresser to a pickle file, `saves.p`, so it can
be accessed later

##### Infrastructure files

* downloader.py: This scrapes the data all 5.7 million pitches in the last 8 years
from the MLB website. It parses HTML to navigate through pages for each month and day to 
determine which games were played on which days and then parses the final XML that contains
the actual game data with all the pitches thrown in the game. It then stores this information
in a mongoDB database in the pitches collection.

* download_batter_stats.py: This scrapes from the same website as downloader.py to get information
on each batter. We ran this on every year from 2008 to 2015. It parses HTML to look find the batter
ID's and then uses to find the url containing the xml data on the batter's season statistics. This information
gets stored in mongoDB in the players collection.

* update_prev_pitch.py: This iterates through all the already-downloaded pitches in our database
and looks up information on the previous pitch, then stores that information directly in the 
current pitch, so that it will be more easily accessible by our feature extracter.

##### Files for plotting results

* plotter.py: This uses matplotlib to plot various figures showing our results. It plots based
on the saved pickle files created by running `grouped_batters.py` and `player_specific.py`

* player_specific.py: This files looks through the database to obtain information about a specific
pitcher. It determines thier average pitch characteristics and counts how often they use particular
pitch sequences. This information is saved in a pickle file so it can later be used by `plotter.py`.
Currently, this is set to obtain data about Clayton Kershaw.