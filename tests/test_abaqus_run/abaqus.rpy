# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2024 replay file
# Internal Version: 2023_09_21-21.55.25 RELr426 190762
# Run by nishioka on Sat Feb 28 17:03:18 2026
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(1.36719, 1.36719), width=201.25, 
    height=135.625)
session.viewports['Viewport: 1'].makeCurrent()
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
execfile('../../src/generate_fairing_dataset.py', __main__.__dict__)
#: Mesh seeds: GLOBAL=1000 mm, DEFECT=200 mm (override)
#: A new model database has been created.
#: The model "Model-1" has been created.
session.viewports['Viewport: 1'].setValues(displayedObject=None)
#: Warning: core faces not found (inner=0, outer=0)
#: Warning: BC: Feature creation failed.
#: Warning: Temperature IC skipped: Feature creation failed.
#: Warning: Thermal load skipped: Feature creation failed.
#: The model database has been saved to "/home/nishioka/Payload2026/tests/test_abaqus_run/test_job.cae".
#: Writing INP for job 'test_job'...
#: Running job 'test_job' with patched INP...
#: Job COMPLETED: test_job.odb
print('RT script done')
#: RT script done
