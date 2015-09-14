# Recommender System

## Using Friend Groups to Reduce the Cold-Start Problem

In everyday life, people rely on recommendations from other people by spoken words, reference letters, news reports from the news media, general surveys, travel guides, and so forth [1]. Recommender systems have been developed to address this complex problem of matching users to items of potential interest. As [2] explains, a good recommender system will recommend items that fall into the users interests. A great recommender system will expand a users interests into neighboring areas. Such a system will not only recommend items that fall into the users interests, but will identify new potential interests from which to suggest items. In the music domain, people in the article from [3] are skeptical about the effectiveness of algorithms to recommend songs:
>"However sophisticated the algorithms are, they will not be able to take into account the random ways in which we discover music and this method of filtering music for us to listen to, is limiting, wrote one commenter."

In order to improve these algorithms, researchers need to take into account the many different ways that users get songs recommended to them. One user from [3] said "word of mouth has a much greater effect than some computer generated recommendation." To my knowledge, limited work has investigated music recommendation that takes into account a user’s group of friends. This could be due to the limited amount of that has been made available to researchers. Some researchers [4, 5] have tried to make up for that by creating arbitrary groups in two ways:

1. Random: stochastic groups makes the assumption that users can meet completely randomly.
2. Similarity Threshold: making the assumption that friends share some sort of similarity in music preferences.

The problem is that this is not a true measure of real social networks and we cannot get a sense of the true information-value by making these assumptions.
By using motivation from assumption (2), that friends share some sort of similarity in music preferences, this paper attempts to look at the information value of social networking information (i.e., bi-directional friend relations) in order to solve the cold-start problem.

Read more: [tsourounis-csc412-project.pdf](https://github.com/guess/recommender/blob/master/tsourounis-csc412-project.pdf)


## References

1. Su, X., & Khoshgoftaar, T. M. (2009). A survey of collabo- rative filtering techniques. Advances in artificial intelligence, 2009.

2. Christodoulakis, C. (2014). Design Considerations for an Ef- fective Recommendation Manager in the Context of the IBM LabBook Distributed Collaborative Analytics Platform.

3. The Gaurdian, ”5 Types of Music Discovery”. http://www.theguardian.com/technology/2014/mar/19/music-discovery-spotify-apps-facebook

4. Roy, S. B., Lakshmanan, L. V., & Liu, R. (2015). From Group Recommendations to Group Formation. arXiv preprint arXiv:1503.03753.

5. Jameson, A. (2004, May). More than the sum of its members: challenges for group recommender systems. In Proceedings of the working conference on Advanced visual interfaces (pp. 48- 54). ACM.
