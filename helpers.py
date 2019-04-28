
_country_indices = {
	'UK': 0,
	'FR': 1,
	'IT': 2,
	'DE': 3,
	'ES': 4,
	'US': 5,
	'??': 6
}

_country_colors = {
	'UK': '#ff0000',
	'FR': '#0000ff',
	'IT': '#00ff00',
	'DE': '#000000',
	'ES': '#ffff00',
	'US': '#00ffff',
	'??': '#0eece6'
}

SECONDS_PER_DAY = 86400
SECONDS_PER_HOUR = 3600
SECONDS_PER_HOUR_F = float(SECONDS_PER_HOUR)



def preprocess_json_for_pandas(json_to_process):
    # The json seems to be formated with lists on every value entry
    # even when these entries are singleton values.
    # Checks this assumptions and formats json in place
    for i in range(len(json_to_process)):
        if len(json_to_process[i]['session_id']) == 1:
            json_to_process[i]['session_id'] = json_to_process[i]['session_id'][0]
        else:
            raise Exception('Data does not follow assumptions: Session ids are multiple')
        if len(json_to_process[i]['unix_timestamp']) == 1:
            json_to_process[i]['unix_timestamp'] = json_to_process[i]['unix_timestamp'][0]
        else:
            raise Exception('Data does not follow assumptions: timestamps are multiple')
        if len(json_to_process[i]['user']) == 1 and len(json_to_process[i]['user'][0]) == 1:
            json_to_process[i]['user'] = json_to_process[i]['user'][0][0]
        else:
            raise Exception('Data does not follow assumptions: users are multiple')


def generate_plot_base():
	# Compute areas and colors
	N = 150
	r = 2 * np.random.rand(N)
	theta = 2 * np.pi * np.random.rand(N)
	area = 200 * r**2
	colors = theta

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='polar')
	c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)

def get_index_for_country(country):
	return _country_indices.get(country,-1)

def get_color_for_country(country):
	return _country_colors.get(country,'#0eece6')

