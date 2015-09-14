from itertools import combinations
import numpy as np


def get_friends(f_name, set_users=None):
    friends = {}
    with open(f_name, 'r') as f:
        next(f)
        for line in f:
            user1, user2 = line.strip().split('\t')
            if set_users is None or (user1 in set_users and user2 in set_users):
                friends = _add_friend(user1, user2, friends)
                friends = _add_friend(user2, user1, friends)
    return friends


def _add_friend(user, friend, friend_dict):
    if user in friend_dict:
        friend_dict[user].add(friend)
    else:
        friend_dict[user] = set([friend])
    return friend_dict


def find_cliques(friends, k=4):
    cliques = {}
    for user in friends.keys():
        # print "{}/{}".format(user, len(friends.keys())-1)
        groups = combinations(friends[user], k)
        for group in groups:
            if _is_clique(group, friends):
                if user in cliques:
                    cliques[user].add(group)
                else:
                    cliques[user] = set([group])
    return cliques


def _is_clique(group, friends):
    num_members = len(group)
    for i in range(num_members):
        for j in range(i+1, num_members):
            if group[i] not in friends[group[j]]:
                return False
    return True


def best_clique(user, cliques, user_sim):
    best_score = -1
    best = None
    for clique in cliques:
        score = np.sum([user_sim[user, friend] for friend in clique])
        if best_score < 0 or score > best_score:
            best_score = score
            best = clique
    return best
