import matplotlib.pyplot as plt
from IPython import display
import pandas as pd
from datetime import datetime
import os

plt.ion()

def plot(scores, mean_scores, collision, collision_point):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, color='b', label="scores")
    plt.plot(mean_scores, color='g', label="mean scores")
    plt.plot(collision, color='r', label='collisions')
    plt.legend(loc="upper left")
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.text(len(collision)-1, collision[-1], str(collision[-1]))
    plt.show(block=False)
    plt.pause(.1)
    
    
    now = datetime.now()
    image_folder_path = f'./data/{now.year}{now.month}{now.day}/image/'
    csv_folder_path = f'./data/{now.year}{now.month}{now.day}/csv/'
    
    if not os.path.exists(image_folder_path):
            os.makedirs(image_folder_path)

    if not os.path.exists(csv_folder_path):
        os.makedirs(csv_folder_path)

    # Salva Imagem
    plt.savefig(os.path.join(image_folder_path, f'{now}.png'))

    # Salva os dados em um arquivo CSV
    data = {'Scores': scores, 'Mean Scores': mean_scores, 'Collision': collision, 'collision_point': collision_point}
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.transpose()
    df.to_csv(os.path.join(csv_folder_path, f'{now}.csv'), index=False)

