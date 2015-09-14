import copy
import random
import numpy as np


def artist_to_count(a2i, f_name):
    ''' Return a dict holding the number of users that like an artist. '''
    a2c = dict()
    with open(f_name, 'r') as f:
        next(f)
        for line in f:
            _, artist, _ = line.strip().split('\t')
            artist_id = a2i[artist]
            if artist_id in a2c:
                a2c[artist_id] += 1
            else:
                a2c[artist_id] = 1
    return a2c


def get_artists(f_name):
    ''' Get the list of artists (items). '''
    artists = []
    with open(f_name, 'r') as f:
        next(f)     # skip the header
        for line in f:
            artist_info = line.strip().split('\t')
            artists.append(artist_info[0])
    return artists


def get_users(f_name):
    ''' Get the list of users. '''
    users = []
    with open(f_name, 'r') as f:
        next(f)
        for line in f:
            user, _, _ = line.strip().split('\t')
            if user not in users:
                users.append(user)
    return users


def convert_to_ind(items):
    ''' Convert a list of items to a dictionary of indices. '''
    item2index = dict()
    for i, item in enumerate(items):
        item2index[item] = i
    return item2index


def user_to_artists(a2i, f_name, set_users=None):
    ''' Get a dictionary containing every user's artist preferences. '''
    pref = dict()
    with open(f_name, 'r') as f:
        next(f)
        for line in f:
            user, artist, _ = line.strip().split('\t')
            artist_id = a2i[artist]
            if set_users is None or user in set_users:
                if user in pref:
                    pref[user].add(artist_id)
                else:
                    pref[user] = set([artist_id])
    return pref


def remove_half_pref(pref):
    ''' Remove half of the preferences in 'pref' for the specified 'users'. '''
    new_pref = copy.deepcopy(pref)
    for user in new_pref.keys():
        num_prefs = len(new_pref[user])
        for _ in xrange(num_prefs / 2):
            new_pref[user].pop()
    return new_pref


def remove_pref(pref, num_to_remove):
    new_pref = copy.deepcopy(pref)
    for user in new_pref.keys():
        for _ in xrange(num_to_remove):
            if len(new_pref[user]) <= 0:
                break
            new_pref[user].pop()
    return new_pref


def artist_to_users(a2i, f_name, set_users=None, ratio=1.0):
    a2u = dict()
    with open(f_name, 'r') as f:
        next(f)
        for line in f:
            if random.random() < ratio:
                user, artist, _ = line.strip().split('\t')
                if set_users is None or user in set_users:
                    artist_id = a2i[artist]
                    if artist_id in a2u:
                        a2u[artist_id].add(user)
                    else:
                        a2u[artist_id] = set([user])
    return a2u


def create_rating_matrix(num_users, num_items, pref):
    ratings = np.zeros(shape=(num_users, num_items))
    for user in pref.keys():
        for item in pref[user]:
            ratings[user, item] = 1
    return ratings


def sort_dict_dec(d):
    return sorted(d.keys(), key=lambda s: d[s], reverse=True)


def get_mpe(pred, target_pref, user_v):
    mpe = 0.0
    for user_i in range(len(pred)):
        user = user_v[user_i]
        num_to_predict = len(target_pref[user]) / 2
        prediction = set(pred[user_i])
        mpe += num_to_predict - len(prediction & target_pref[user])
    mpe /= len(pred)
    return mpe
