from sklearn.linear_model import Ridge as Regressor
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import dateutil.parser as parser
from datetime import datetime

def load_file(file_name):
    malformed = 0
    lines = list(open(file_name))
    col1, col2 = lines[0].strip().split(',')
#    print(col1, col2, len(lines))
    lines = lines[1:]
    data = []
    for line in lines:
        line = line.strip()
        date_string, usage_string = line.split(',')
        timestamp = parser.parse(date_string, dayfirst=False)
        usage_string = usage_string.strip()
        if not usage_string:
            malformed += 1
            continue
        usage = float(usage_string)
        data.append((timestamp, usage))
    return data, malformed

def partition_data(data):
    training_percent = 0.92
    n = len(data)
    split = int(n * training_percent)
    return data[:split], data[split:]

def encode_month(month):
    zeros = [0 for _ in range(12)]
    zeros[month - 1] = 1
    return zeros

def encode_day(day):
    return [day]

def encode_hour(hour):
    zeros = [0 for _ in range(24)]
    zeros[hour - 1] = 0
    return zeros

def encode_minute(minute):
    i = int(minute / 15)
    zeros = [0 for _ in range(4)]
    zeros[i] = 1
    return zeros

def encode_trend(time):
    weeks = int(time / (60 * 60 * 24 * 7)) - 2485
    return [weeks, weeks**3]

def multiply_features(l1, l2):
    l1 = l1 if l1 else []
    l2 = l2 if l2 else []
    result = []
    for x in [1.] + l1:
        for y in [1.] + l2:
            result.append(x * y)
    return result

def featurize_time(dt):
    date = dt.date()
    time = dt.time()
    features = []
    features.extend(encode_month(date.month))
    features = multiply_features(features, encode_day(date.day))
    features = multiply_features(features, encode_hour(time.hour))
    features = multiply_features(features, encode_minute(time.minute))
    features = multiply_features(features, encode_trend(dt.timestamp()))
    return features

def separate(data):
    times, values = [], []
    for time, value in data:
        times.append(time)
        values.append(value)
    return times, values

def to_model_params(data):
    print("Featurizing...")
    times, usages = separate(data)
    features = [featurize_time(t) for t in times]
    print("Done.")
#    f = np.array(features)
#    pf = PolynomialFeatures(degree=2)
#    X = pf.fit_transform(f)
    X = np.array(features)
    y = np.array(usages).reshape(-1, 1)
    return X, y

def train(data, lamb = 1.0):
    X, y = to_model_params(data)
    return train_params(X, y, lamb = lamb)

def train_params(X, y, lamb = 1.0):
    model = Regressor(alpha=lamb)
    model.fit(X, y)
    return model

def calculate_error(model, data):
    X, y = to_model_params(data)
    return calculate_error_params(model, X, y)

def calculate_error_params(model, X, y):
    y_hat = model.predict(X)
    return np.linalg.norm(y_hat - y) / len(data)

data, errors = load_file('data.csv')
#print('Number of lines successfully parsed %i, failed %i' % (len(data), errors))
training, validation = partition_data(data)
#
#validation_error = []
X, y = to_model_params(data)

#for l in np.arange(0, 1000, 50):
#    model = train_params(X, y, lamb = l)
#    print('-'*6 + str(l) + '-'*6)
#    train_error = calculate_error_params(model, X, y) 
#    valid_error = calculate_error_params(model, X, y) 
#    validation_error.append((l, valid_error))
#    print('Training error: %f' % train_error)
#    print('Validate error: %f' % valid_error)

#for x, y in validation_error:
#    print('%f, %f' % (x, y))
#train_x, train_y = separate(training)
#valid_x, valid_y = separate(validation)

model = train_params(X, y, lamb=500)

times = []
t = 1514764800 + 8 * 60 * 60 
dt = datetime.fromtimestamp(t)
#print(dt)
while dt.date().month < 2:
    times.append(dt)
    t += 60 * 15
    dt = datetime.fromtimestamp(t)

features = [featurize_time(dt) for dt in times]
X = np.array(features)
y = model.predict(X)
csv = open('predictions.csv', 'w')
for dt, y, in zip(times, list(y.reshape(-1))):
    csv.write('%s, %f\n' % (dt.strftime('%Y-%m-%d %H:%M:%S'), y))
    print(dt.strftime('%Y-%m-%d %H:%M:%S'), y)

csv.close()
