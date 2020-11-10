
class Evaluation():
    def __init__(self, of_any_instance_seeds=(0, ""),
                       of_any_instance_class=(0, ""),
                       types_individual=[],
                       types_collective=[],
                       num_truth_pixels=0,
                       name=''):
        self.of_any_instance_from_seeds = of_any_instance_seeds  # binary "yes or no" this pixel is in any instance, purely from seedmap
        self.of_any_instance_class = of_any_instance_class # binary "yes or no" this pixel is in any instance, from the post-processing process
        self.types_individual = types_individual  # Just looking at each type on its own, comparing it to 
                                                  # that type in the truth
        self.types_collective = types_collective  # Looking at each type, but ensuring there's no overlap 
                                                  # across types, i.e. one pixel cannot be part of two types
        
        self.num_truth_pixels = 0
        self.location_name = name
    
    def average(self):
        return (self.of_any_instance_from_seeds[0] + self.of_any_instance_class[0] + \
               sum([elem[0] for elem in self.types_individual]) + \
               sum([elem[0] for elem in self.types_collective])) / float(14)

    def __str__(self):
        return "{} {} {} {} {} {}".format(self.of_any_instance_from_seeds,
                                          self.of_any_instance_class,
                                          self.types_individual,
                                          self.types_collective,
                                          self.num_truth_pixels,
                                          self.location_name)