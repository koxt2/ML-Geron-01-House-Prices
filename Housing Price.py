import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tarfile
import urllib.request
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import time

start_time = time.time()

########## To plot figures ##########
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

########## Where to save the figures ##########
PROJECT_ROOT_DIR = ""
CHAPTER_ID = "housing_price_model"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

########## Import the house price data ##########
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = PROJECT_ROOT_DIR + "dataset"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

########## Load the data ##########
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

fetch_housing_data()

########## Summarise/analyse the data ##########
housing = load_housing_data() 
print(housing) # Prints the housing data
print(housing.info()) # Prints a summary of the housing data
print(housing["ocean_proximity"].value_counts()) # Prints a count of values of one of the columns
print(housing.describe()) # Prints a collection of averages

housing.hist(bins=50, figsize=(20,15)) # Creates a collection of histograms
save_fig("attribute_histogram_plots")

########## Create a train set and test set (80/20) - stratify the sample based on median income ##########
# Have a look at the data that is going to be the base strata
housing["income_cat"] = pd.cut(housing["median_income"], # Group the coloumn data into categories, ie 0-£15k, £15-30k etc
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

housing["income_cat"].hist()
print(housing["income_cat"].value_counts()) # Print the count of each income category

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print(len(strat_train_set)) # Check that the sum of the two new datasets equal the number of the original set
print(len(strat_test_set))

########## Compare stratified test set proportions to original sample ##########
print(housing["income_cat"].value_counts() / len(housing)) # Original sample
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set)) # Test set

########## Create a table to compare proportions ##########
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
print(compare_props)

########## Create a scatter diagram of the coordinates and colour code by house value ##########
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
plt.legend()
save_fig("housing_prices_scatterplot")

########## Return the strat data back to it's original form (drop the income category) ##########
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

########## Overlay diagram onto a map ##########
# Download the California image
images_path = IMAGES_PATH
os.makedirs(images_path, exist_ok=True)
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
filename = "california.png"
print("Downloading", filename)
url = DOWNLOAD_ROOT + "images/end_to_end_project/" + filename
urllib.request.urlretrieve(url, os.path.join(images_path, filename))

california_img=mpimg.imread(os.path.join(images_path, filename))
ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                  s=housing['population']/100, label="Population",
                  c="median_house_value", cmap=plt.get_cmap("jet"),
                  colorbar=False, alpha=0.4)
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
           cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar(ticks=tick_values/prices.max())
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
save_fig("california_housing_prices_plot")

########## Look for correlations ##########
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

########## Create scatter diagrams of the main attributes ##########
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
save_fig("scatter_matrix_plot")

########## Create scatter diagram of the strongest correlation ##########
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
plt.axis([0, 16, 0, 550000])
save_fig("income_vs_house_value_scatterplot")

########## Look into some other correlations ##########
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"] 
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

########## Prepare data for machine learning algorithm ##########
# Revert to a clean training set
housing = strat_train_set.drop("median_house_value", axis=1) # Makes a copy of strat_train_set but without median_house_value
housing_labels = strat_train_set["median_house_value"].copy() # Copies just the median_house_value from strat_train_set

# Note; from earlier summary that total number of bedrooms only has a figure for 20433 samples from a total pool of 20640
# Use sklearn.impute to fill null rows with median
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1) # imputer only works on numerical data and 'ocean proximity' is text, so drop it. 
imputer.fit(housing_num) # Calculate the median and store the result

# In case any new data has missing values use imputer to calculate the median of all the other columns
imputer.statistics_

housing_num.median().values # manually calculate the median to check imputer did it correctly... check the results are the same
print(imputer.statistics_)
print(housing_num.median().values)

X = imputer.transform(housing_num) # add the calculated median (imputer.fit) to the data sample

########## Covert text data to numerical ##########
# A look at the first 10 rows of the alpha data
housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(10))

# First method - OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:10]) # Print first 10 rows of encoded data
print(ordinal_encoder.categories_) # Print a list of the categories
# ML algorithms will assume two nearby values are more similar that two distant values

# Second method - OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

########## Create a custom transformer to add extra attributes ##########
# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

########## Create a pipeline for preprocessing numerical attributes ##########
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

########## Apply transformations to all columns ##########
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)

########## LINEAR REGRESSION ##########   

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("""


**********     Linear Regression     **********""")
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))

########## Test the model against the whole test sample ##########
# Result is the prediction error $
housing_predictions = lin_reg.predict(housing_prepared)
# Mean squared error
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
# mean absolute error
lin_mae = mean_absolute_error(housing_labels, housing_predictions)

print("Lin_RMSE", lin_rmse)
print("Lin_MAE", lin_mae)

########## Tune using cross validation ##########
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)

print(display_scores(lin_rmse_scores))

########## DECISION TREE REGRESSOR ##########
print("""


**********     Decision Tree Regression     **********""")
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)

print(tree_rmse)

########## Tune model using cross validation ##########
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

print(display_scores(tree_rmse_scores))

########## RANDOM FOREST REGRESSOR ##########
print("""


**********     Random Forest Regressor     **********""")
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)

print(forest_rmse)

########## Tune using cross validation ##########
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
print(display_scores(forest_rmse_scores))

########## Random forest regressor seems most accurate - fine tune further ##########
# Use GridSearch to play with the hyperparameters
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

print(grid_search.fit(housing_prepared, housing_labels))
print(grid_search.best_params_) # Prints the best hyperparameter combination
print(grid_search.best_estimator_) # Also prints the best hyperparameter combination

cvres = grid_search.cv_results_ # Print results
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

########## Evaluate final model on test set ##########
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print("Final Random Tree Regressor RMSE - ", final_rmse)

print ("My program took", time.time() - start_time, "to run")
 























