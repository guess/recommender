from sklearn import cross_validation
import numpy as np
import sys
import util
import clique
import rec


# DATA FILES
f_user_artists = "data/user_artists.dat"
f_artists = "data/artists.dat"
f_friends = "data/user_friends.dat"


print 'loading artists in {}'.format(f_artists)
sys.stdout.flush()
artists = util.get_artists(f_artists)
a2i = util.convert_to_ind(artists)


print 'default ordering by popularity'
sys.stdout.flush()
artists_ordered = util.sort_dict_dec(util.artist_to_count(a2i, f_user_artists))


print 'loading all users in {}'.format(f_user_artists)
sys.stdout.flush()
users = np.array(util.get_users(f_user_artists))
# u2i = util.convert_to_ind(users)


print 'creating cross-validation splits'
sys.stdout.flush()
user_split = cross_validation.ShuffleSplit(
    len(users), 1, test_size=0.25, random_state=0)


for t_ind, v_ind in user_split:

    print 'loading training users and converting to indices'
    sys.stdout.flush()
    users_t = users[t_ind]
    u2i = util.convert_to_ind(users_t)

    print 'artist to users on {}'.format(f_user_artists)
    sys.stdout.flush()
    a2u_tr = util.artist_to_users(a2i, f_user_artists, set_users=users_t)

    print 'converting users to indices'
    sys.stdout.flush()
    for artist in a2u_tr:
        a_set = set()
        for u in a2u_tr[artist]:
            a_set.add(u2i[u])
        a2u_tr[artist] = a_set

    print 'user to artists on {}'.format(f_user_artists)
    sys.stdout.flush()
    u2a = util.user_to_artists(a2i, f_user_artists, set_users=users_t)

    print 'converting users to indices'
    sys.stdout.flush()
    u2a_tr = {}
    for user in u2a_tr:
        u_set = set()
        for artist in u2a[user]:
            u_set.add(artist)
        u2a_tr[u2i[user]] = u_set

    #del u2i
    del u2a

    print 'loading validation users'
    sys.stdout.flush()
    users_v = users[v_ind]

    print 'loading user preferences in validation set'
    sys.stdout.flush()
    u2a_target = util.user_to_artists(a2i, f_user_artists, set_users=users_v)
    # u2a_v = util.remove_half_pref(u2a_target)
    # u2a_v = util.remove_pref(u2a_target, 10)

    print 'loading friend relations'
    friends = clique.get_friends(f_friends)
    cliques = clique.find_cliques(friends, 2)

    # NOTE: Optimal _A=0 and _Q=1 for both
    _A = 0
    _Q = 1

    for i in [40]:

        # Hide i preferences
        u2a_v = util.remove_pref(u2a_target, i)

        # for user in users_v:
        #     if user in cliques and len(cliques[user]) > 0:
        #         c = cliques[user].pop()
        #         for u in c:
        #             if u not in u2i or u2i[u] not in u2a_tr:
        #                 continue
        #             for a in u2a_tr[u2i[u]]:
        #                 u2a_v[user].add(a)

        # print 'creating item predictor...'
        # sys.stdout.flush()
        pr_item = rec.PredSI(a2u_tr, _A, _Q)

        # print 'creating user predictor...'
        # sys.stdout.flush()
        # pr_user = rec.PredSU(u2a_tr, _A, _Q)

        # print 'creating recommender'
        cp = rec.SReco(artists_ordered)
        cp.Add(pr_item)
        # cp.Add(pr_user)
        cp.Gamma = [1.0]

        r = cp.RecommendToUsers(users_v, u2a_v)

        print "mAP@100[{}]: {}".format(i, rec.mAP(users_v, r, u2a_target, 100))

    # print "MPE: {}".format(util.get_mpe(r, u2a_target, users_v))
