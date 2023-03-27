# Procedure for QA generation from annotations

## Preprocessing the annotations to store variables needed in templates:
- store_processed_data_in_annotation.ipynb: takes annotation folders of scatter, line, horizontal bars and vertical bars as input, calculate different variables as number of major ticks, names of major ticks, number of data series, etc., and store them as task 7, creating a copy of annotations in a different folder.
- preprocessing_vertical_box.ipynb: takes annotation folders of vertical boxplots as input, calculate different variables as number of major ticks, names of major ticks, number of data series, etc., and store them as task 7, creating a copy of annotations in a different folder.

## Creating QA pairs for all charts
(Take template for QAs in parent directory)
- QA_gen.ipynb: creating QA pairs for scatter, line, horizontal bars and vertical bars using the preprocessed annotations created above.
- QA_gen_box.ipynb: creating QA pairs for vertical box plots using the preprocessed annotations created above.

### extrapolate.ipynb is a piece of code that extrapolates the dependent value of chart at major axis (if not annotated), taking 2 points which are at nearest left and nearest right of major axis.
