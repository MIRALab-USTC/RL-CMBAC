"""
copy from rlkit
Common evaluation utilities.
"""
import collections
from collections import OrderedDict
from numbers import Number
import numpy as np




def get_generic_path_information(paths, stat_prefix='', report_final_initial=False):
    """
    Get an OrderedDict with a bunch of statistic names and values.
    """

    statistics = OrderedDict()
    returns = [np.sum(path["rewards"]) for path in paths]

    rewards = np.vstack([path["rewards"] for path in paths])
    statistics.update(create_stats_ordered_dict('Rewards', rewards,
                                                stat_prefix=stat_prefix))
    statistics.update(create_stats_ordered_dict('Returns', returns,
                                                stat_prefix=stat_prefix))
    statistics['Num Paths'] = len(paths)
    statistics[stat_prefix + 'Average Returns'] = get_average_returns(paths)

    for info_name in ['env_infos', 'agent_infos']:
        if info_name in paths[0]:
            keys = paths[0][info_name].keys()
            for k in keys:
                if report_final_initial:
                    final_ks = paths[0][info_name][k]
                    first_ks = paths[-1][info_name][k]
                    statistics.update(create_stats_ordered_dict(
                        stat_prefix + k,
                        final_ks,
                        stat_prefix='{}/final/'.format(info_name),
                    ))
                    statistics.update(create_stats_ordered_dict(
                        stat_prefix + k,
                        first_ks,
                        stat_prefix='{}/initial/'.format(info_name),
                    ))

                all_ks = np.concatenate([path[info_name][k] for path in paths])
                statistics.update(create_stats_ordered_dict(
                    stat_prefix + k,
                    all_ks,
                    stat_prefix='{}/'.format(info_name),
                ))

    return statistics


def get_average_returns(paths):
    returns = [sum(path["rewards"]) for path in paths]
    return np.mean(returns)


def create_stats_ordered_dict(
        name,
        data,
        stat_prefix=None,
        always_show_all_stats=True,
        exclude_max_min=False,
):
    if stat_prefix is not None:
        name = "{}{}".format(stat_prefix, name)
    if isinstance(data, Number):
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if (isinstance(data, np.ndarray) and data.size == 1
            and not always_show_all_stats):
        return OrderedDict({name: float(data)})

    stats = OrderedDict([
        (name + ' Mean', np.mean(data)),
        (name + ' Std', np.std(data)),
    ])
    if not exclude_max_min:
        stats[name + ' Max'] = np.max(data)
        stats[name + ' Min'] = np.min(data)
    return stats
