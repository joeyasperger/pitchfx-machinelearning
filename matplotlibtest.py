import matplotlib.pyplot as plt
import numpy as np


# def color(num):
#     val = (num - .45) * 4
#     return [min(.8, 1-val), 0, val]


# grid = [
#     [color(.70),color(.60),color(.60)],
#     [color(.55),color(.45),color(.58)],
#     [color(.70),color(.60),color(.70)]
# ]

# fig, axes = plt.subplots(3, 3, figsize=(8, 6),
#                          subplot_kw={'xticks': [], 'yticks': []})

# fig.subplots_adjust(hspace=0.3, wspace=0.05)

# for ax in axes.flat:
#     ax.imshow(grid, interpolation='none', cmap='brg')
#     ax.set_title('title')

# plt.plot()
# fig = plt.figure() 
# fig.canvas.set_window_title('asdfasd') 
# grid = [
#     [color(.70),color(.60),color(.60)],
#     [color(.55),color(.75),color(.58)],
#     [color(.70),color(.60),color(.70)]
# ]

# fig, axes = plt.subplots(1, 3, figsize=(8, 6),
#                          subplot_kw={'xticks': [], 'yticks': []})

# fig.subplots_adjust(hspace=0.3, wspace=0.05)

# for ax in axes.flat:
#     ax.imshow(grid, interpolation='none', cmap='brg')
#     ax.set_title('title')


# plt.plot()
# plt.show()


N = 9
pitches = (.7, .45, .5, .6, .7, .65, .55, .5, .62)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects = ax.bar(ind, pitches, width, color='r')


# add some text for labels, title and axes ticks
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('TL', 'TM', 'TR', 'ML', 'MM', 'MR', 'BL', 'BM', 'BR') )

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%.2f'% height,
                ha='center', va='bottom')

autolabel(rects)

plt.show()