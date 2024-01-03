
import pandas as pd 
import lightgbm as lgbm
import seaborn as sns
import matplotlib.pyplot as plt
def show_feat_importance(df):
    TOP = 150
    importance_data = pd.DataFrame({'name': train.drop('target', axis= 1).columns
    , 'importance': lgbm.feature_importances_})
    importance_data = importance_data.sort_values(by='importance', ascending=False)

    fig, ax = plt.subplots(figsize=(25,25))
    sns.barplot(data=importance_data[:TOP],
                x = 'importance',
                y = 'name'
            )
    patches = ax.patches
    count = 0
    for patch in patches:
        height = patch.get_height() 
        width = patch.get_width()
        perc = 0.01*importance_data['importance'].iloc[count]#100*width/len(importance_data)
        ax.text(width, patch.get_y() + height/2, f'{perc:.1f}%')
        count+=1

    plt.title(f'The top {TOP} features sorted by importance')
    plt.show()