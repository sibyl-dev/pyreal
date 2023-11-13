import cProfile
import pstats
from pyreal.sample_applications import ames_housing

# Setup
app = ames_housing.load_app()
x = ames_housing.load_data()

profiler = cProfile.Profile()
profiler.enable()

# Analysis code
app.produce_feature_contributions(x)
profiler.disable()

stats = pstats.Stats(profiler)
# this is the only way I can figure out to filter only package function calls
stats.sort_stats("tottime").print_stats("github")
