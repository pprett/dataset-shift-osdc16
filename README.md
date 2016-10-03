# Dataset Shift in Machine Learning

Supporting material for my OSDC London'16 talk on Dataset Shift in Machine Learning

## Checkerboard

<img src="../master/.assets/checkerboard.png?row=true" width="200">

Interactive plot to illustrate the phenomenon of covariate shift.
That is, p(x,y) differs from training to testing phase. The interactive plot allows you
to specify p(x) via a probability tabels while p(y|x) remains fixed.
You can see that underspecified discriminative models are not immune to covariate shift.

Use the ``bokeh serve`` command to run the interactive plot:

    bokeh serve bokeh_checkerboard.py

at your command prompt. Then navigate to the URL

    http://localhost:5006/bokeh_checkerboard

in your browser .