import matplotlib.pyplot as plt

colors = {}

def visualization_setup(particle_names):
    colormap = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
        '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3',
        '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']

    keys = particle_names.keys()
    keys.sort()
    for idx, key in enumerate(keys):
        colors[key] = colormap[idx%len(colormap)]


def visualize(entry, coord_dict, events, IMG_WIDTH, IMG_HEIGHT, IMG_BUFF, vis, vis_uneven, instances_uneven, particle_names):
    for plane, coord_t in coord_dict.items(): # random for each plane
        if ((vis != None) and (random.random()*100 < vis)) or \
            (vis_uneven and instances_uneven):

            x, y, dummy = zip(*coord_t)
            plt.ylim(0, IMG_WIDTH + IMG_BUFF)
            plt.xlim(0, IMG_HEIGHT + IMG_BUFF)
            plt.title("Event {}, plane {}".format(events, plane))

            plt.plot(x, y, '.', markersize=5, color='black')
            for inst_idx in xrange(entry.DataBranch.num_instances_plane(plane)):
                inst_xs, inst_ys = zip(*entry.DataBranch.instance(plane, inst_idx))
                type_inst = abs(entry.DataBranch.typeof_instance(plane, inst_idx))
                plt.plot(inst_xs, inst_ys, '.', markersize=7, color=colors[type_inst], label=particle_names[type_inst])
            plt.legend()
            plt.show()
