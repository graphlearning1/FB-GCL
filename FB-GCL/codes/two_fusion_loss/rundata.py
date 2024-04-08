import os
import subprocess as sp

datasets = ["photo", "pubmed", "computer"]
for data in datasets:
    path = data
    with open('../results/abl/{}_res666.txt'.format(data), 'wb') as out:
            sp.call(['python', 'main_fusion_adpabl.py',
                     '--dataname', data
                     ], stdout=out)




datasets = ["photo", "cs", "pubmed", "computer"]
for data in datasets:
    path = data
    with open('../results/raw/{}_res666.txt'.format(data), 'wb') as out:
            sp.call(['python', 'main_fusion.py',
                     '--dataname', data
                     ], stdout=out)

datasets = ["physics"]
for data in datasets:
    path = data
    with open('../results/abl/{}_res666.txt'.format(data), 'wb') as out:
            sp.call(['python', 'main_fusion_physics_adpabl.py',
                     '--dataname', data
                     ], stdout=out)

datasets = ["physics"]
for data in datasets:
    path = data
    with open('../results/raw/{}_res666.txt'.format(data), 'wb') as out:
            sp.call(['python', 'main_fusion.py',
                     '--dataname', data
                     ], stdout=out)