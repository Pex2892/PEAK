from dataset import Dataset
from eda import EDA
from correlation import Correlation
from regression import Regression
from classifier import Classifier
import utilities as ut

# ----- INIT -----
ut.header()

# ----- READ ARGUMENTS -----
args = ut.read_args()

# ----- CLEAR PREVIOUS RESULTS -----
ut.clear_data(args['settings']['clear'])

# ----- LOAD DATASET -----
ds = Dataset(settings=args['dataset'])

# ----- REMOVED COLUMNS -----
ds.drop_cols()

# ----- REMOVED ROWS -----
ds.drop_rows()

# ----- NORMALIZED COLUMNS -----
ds.normalize()

# ----- COVERT COLUMNS FROM CATEGORIAL TO NUMERIC -----
ds.categorial_to_numeric()

# ----- DATASET BACKUP -----
ds.export_csv()

# ----- EXPLORATORY DATA ANALYSIS -----
eda = EDA()
eda.automate(ds)

# ----- CORRELATION -----
corr = Correlation()
corr.automate(ds)

# ----- LINEAR REGRESSION -----
lm = Regression(ds, args['regression'], args['settings'])
lm.automate()

# ----- CLASSIFIERS -----
clf = Classifier(ds, args['classification'], args['settings'])
clf.automate()

