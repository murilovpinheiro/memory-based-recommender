import numpy as np
import scipy.sparse as sps
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
import random


class DistanceCollaborativeRecommender:
    """
    Distance Collaborative Recommender Class.

    Parameters:
    - data: Scipy Sparse Matrix (CSR for Item recommendation, CSC for User recommendation).
    - itemNames: pandas Series of strings or integers representing item names.
    - userNames: pandas Series of strings or integers representing user names.
    - type_of_recommendation: Type of recommendation, "Item" or "User".
    - metric: Distance Metric in recommendation, "pearson" or "cosine"

    Raises:
    - ValueError: If data is not a Scipy Sparse Matrix, itemNames/userNames are not Series of strings or integers,
                  type_of_recommendation is not "Item" or "User",
                  or if the shape of data is not appropriate for the recommendation type.
    """

    def __init__(self, data, itemNames=None, userNames=None, type_of_recommendation="Item", metric="pearson"):
        """
        Initialize the DistanceCollaborativeRecommender.

        Parameters:
        - data (sps.csr_matrix or sps.csc_matrix): Scipy Sparse Matrix (CSR for Item recommendation, CSC for User recommendation).
        - itemNames (pandas Series, optional): Series of strings or integers representing item names.
        - userNames (pandas Series, optional): Series of strings or integers representing user names.
        - type_of_recommendation (str, optional): Type of recommendation, "Item" or "User".
        - metric (str, optional): Distance Metric in recommendation, "pearson" or "cosine".

        Raises:
        - ValueError: If data is not a Scipy Sparse Matrix, itemNames/userNames are not Series of strings or integers,
                      type_of_recommendation is not "Item" or "User",
                      or if the shape of data is not appropriate for the recommendation type.
        """
        if not sps.issparse(data):
            raise ValueError("The 'data' parameter must be a Scipy Sparse Matrix (CSR or CSC).")

        if type_of_recommendation not in ["Item", "User"]:
            raise ValueError("The 'type_of_recommendation' parameter must be 'Item' or 'User'.")

        if (type_of_recommendation == "Item" and not sps.isspmatrix_csc(data)) or (
                type_of_recommendation == "User" and not sps.isspmatrix_csr(data)):
            raise ValueError(
                f"For 'type_of_recommendation' equal to '{type_of_recommendation}', 'data' must be a matrix of type {type_of_recommendation}.")

        if data.shape[1] > data.shape[0]:
            raise ValueError(
                "The number of rows should be for 'Users', and the number of columns should be for 'Items'.")

        if metric.lower() != "pearson" and metric.lower() != "cosine":
            raise ValueError("The Distance metric must be pearson or cosine.")

        if userNames is None:
            userNames = pd.Series(np.arange(data.shape[0]))
        elif len(userNames) != data.shape[0]:
            raise ValueError("The size of the users names list needs to be equals to the number of users.")

        if itemNames is None:
            itemNames = pd.Series(np.arange(data.shape[1]))
        elif len(itemNames) != data.shape[1]:
            raise ValueError("The size of the items names list needs to be equals to the number of items.")

        if not all(isinstance(name, (str, int)) for name in itemNames):
            raise ValueError("The 'itemNames' parameter must be a list containing only strings or integers.")

        if not all(isinstance(name, (str, int)) for name in userNames):
            raise ValueError("The 'userNames' parameter must be a list containing only strings or integers.")

        self.data = data
        self.itemNames = itemNames
        self.userNames = userNames
        self.recm = type_of_recommendation
        self.metric = metric

    def _correlation_pearson_sparse(self, array):
        """
        Calculate Pearson correlation for sparse matrices.

        Parameters:
        - array (np.ndarray): Numpy array containing only numbers.

        Returns:
        - corr (np.ndarray): Pearson correlation values.
        """
        if not isinstance(array, np.ndarray) or not np.issubdtype(array.dtype, np.number):
            raise ValueError("The 'array' parameter must be a numpy array containing only numbers.")

        if self.recm == "User":
            axis = 1
        elif self.recm == "Item":
            axis = 0

        yy = array - array.mean()
        xm = (self.data).mean(axis=axis).A.ravel()
        ys = yy / np.sqrt(np.dot(yy, yy))
        xs = np.sqrt(
            np.add.reduceat((self.data).data ** 2, (self.data).indptr[:-1]) - (self.data).shape[axis] * xm * xm)

        corr = np.add.reduceat((self.data).data * ys[(self.data).indices], (self.data).indptr[:-1]) / xs
        return corr

    def _search_name(self, search, search_type):
        """
        Search for a name in itemNames/userNames.

        Parameters:
        - search (str or int): Name or index to search for.

        Returns:
        - result (str or int or None): Found name or index, or None if not found.
        """
        if search_type == "User":
            list_to_search = self.userNames
        else:
            list_to_search = self.itemNames
        if search in list_to_search:
            return search
        if isinstance(search, str):
            escaped_search = re.escape(search)
            result = (list_to_search.loc[list_to_search.str.contains(escaped_search)]).reset_index(drop=True)
            if result.empty:
                return None
            return result.loc[0]
        return None

    def _get_items_similars(self, item_identification):
        """
        Get item recommendations based on item identification, private to be used by the class.

        Parameters:
        - item_identification (int or str): Item index or name.

        Returns:
        - similarity values.
        """
        item_index = (self.itemNames == item_identification).idxmax()

        movie = self.data[:, item_index].toarray().ravel()

        if self.metric == "cosine":
            similarity = cosine_similarity(movie.reshape(1, -1), self.data.T)[0]
        elif self.metric == "pearson":
            similarity = self._correlation_pearson_sparse(movie)

        # order = np.argsort(-similarity)[1:]

        return similarity

    def _get_recommendation_item(self, user_identification):
        if isinstance(user_identification, (str, np.str_)) or (isinstance(user_identification, int)):
            # Check if the username is in the list and get its index
            found_indices = [index for index, name in enumerate(self.userNames) if name == user_identification]

            if found_indices:
                user_index = found_indices[0]
            else:
                raise ValueError(f"User '{user_identification}' not found.")
        else:
            raise ValueError("Invalid user identification. Provide an integer index or a username.")

        first_user = self.data[user_index, :].toarray().ravel()
        user_items = np.where(first_user > 0.)[0]

        num_columns = self.data.shape[1]

        total = np.zeros(num_columns)
        similarity_sums = np.zeros(num_columns)
        data_csc = self.data.tocsc()  # Need this to optimize in the for
        for col in user_items:
            similarity = self._get_items_similars(self.itemNames[col])
            similarity_sums += similarity * first_user[col]  # similarities of a movie * rating of the same movie

        similarity_sums[user_items] = 0
        similarity_final = similarity_sums

        order = (np.argsort(-similarity_final))

        return self.itemNames[order], similarity_final[order]

    def _get_recommendation_user(self, user_identification):
        """
        Get user recommendations based on user identification.

        Parameters:
        - user_identification (int or str): User index or name.

        Returns:
        - Tuple: Ordered item names and similarity values.
        """
        if (isinstance(user_identification, (str, np.str_)) or (isinstance(user_identification, int))):
            # Check if the username is in the list and get its index
            found_indices = [index for index, name in enumerate(self.userNames) if name == user_identification]

            if found_indices:
                user_index = found_indices[0]
            else:
                raise ValueError(f"User '{user_identification}' not found.")
        else:
            raise ValueError("Invalid user identification. Provide an integer index or a username.")
        first_user = self.data[user_index, :].toarray().ravel()
        user_items_index = np.where(first_user > 0.)[0]
        if self.metric == "cosine":
            similarity_table = cosine_similarity(first_user.reshape(1, -1), self.data)[0]
        elif self.metric == "pearson":
            similarity_table = self._correlation_pearson_sparse(first_user)

        num_columns = (self.data).shape[1]  # take num_movie

        # Creating the two Results for the ranking
        total = np.zeros(num_columns)
        similaritySums = np.zeros(num_columns)
        data_csc = ((self.data).tocsc())  # Need this to optimize in the for

        for col in range(num_columns):
            if col in user_items_index: continue
            column_data = data_csc[:, col].T
            column_sum = column_data.nnz
            # For each unwatched movie we calculate the (Similarity * Rating) Sum and the Similarity Sum 
            total[col] = column_data.dot(similarity_table)

            column_data.data = np.ones(column_data.data.shape)
            similaritySums[col] = (column_data.dot(similarity_table)) / column_sum

        # the division between the total and similaritySums result in the "Predicted" Rate for that movie, calculated using the rate of the observed Rate
        similaritySums_smoothed = similaritySums + 1
        # # to avoid division by 0, for a reason idk when i divide the recommendation is the same for everyone
        similarity_final = (total / similaritySums_smoothed)

        order = (np.argsort(-similarity_final))

        return self.itemNames[order], similarity_final[order]

    def get_random_elements(self, num_elements=1):
        """
        Get random item or user names.

        Parameters:
        - num_elements: Number of random elements to retrieve (default is 1).

        Returns:
        - random_elements: List of random item or user names.
        """
        if num_elements > 10:
            raise ValueError("Num_elements can't be more than 10.")

        if self.recm == "User":
            return random.sample(self.userNames, num_elements)
        else:
            return random.sample(self.itemNames, num_elements)

    def get_recommendation(self, to_recommend, N=10, need_numeric=False):
        """
        Get recommendations based on a name/index.

        Parameters:
        - to_recommend: the Name or Index to calculate the recommendation.
        - N: Number of recommendations to return (default is 10).
        - need_numeric: If True, return only the item IDs; if False, return item names and similarity values (default is False).

        Returns:
        - recommendations: Tuple containing ordered item names and similarity values or item IDs and the search_result.
        """
        search_result = self._search_name(to_recommend, "User")

        if search_result is None:
            print(f"{self.recm} not found.")
            return None

        if self.recm == "User":
            recommendation, values = self._get_recommendation_user(search_result)
        elif self.recm == "Item":
            recommendation, values = self._get_recommendation_item(search_result)

        if need_numeric:
            return (search_result, recommendation[0:N], values[0:N])

        return (search_result, recommendation[0:N])

    def get_items_similars(self, item_identification):
        """
        Get item recommendations based on item identification, public, can be used by the others classes.

        Parameters:
        - item_identification (int or str): Item index or name.

        Returns:
        - Tuple: Ordered item names and similarity values.
        """
        # Adapt item_identification to handle both index and item name
        item_identification = self._search_name(item_identification, "Item")

        if isinstance(item_identification, int):
            item_index = item_identification
        elif isinstance(item_identification, (str, np.str_)):
            # Check if the item name is in the list and get its index
            # I dont know why but if item_id in itemNames dont work
            found_indices = [index for index, name in enumerate(self.itemNames) if name == item_identification]

            if found_indices:
                item_index = found_indices[0]
            else:
                print(f"Item '{item_identification}' not found.")
                raise ValueError("Item not found.")
        else:
            raise ValueError("Invalid item identification. Provide an integer index or an item name.")

        movie = self.data[:, item_index].toarray().ravel()

        if self.metric == "cosine":
            similarity = cosine_similarity(movie.reshape(1, -1), self.data.T)[0]
        elif self.metric == "pearson":
            similarity = self._correlation_pearson_sparse(movie)
        else:
            raise ValueError("Invalid metric. Choose 'cosine' or 'pearson'.")

        # order = np.argsort(-similarity)[1:]

        return self.itemNames, similarity

    def get_user_items(self, user_identification):
        search_result = self._search_name(user_identification, "User")

        if search_result is None:
            print(f"User not found.")
            return None

        user_index = (self.userNames == user_identification).idxmax()

        first_user = self.data[user_index, :].toarray().ravel()
        user_items = first_user > 0
        user_items_names = self.itemNames[user_items]

        # Ordenar user_items_names de acordo com os valores correspondentes em first_user
        sorted_indices = np.argsort(first_user[user_items])[::-1]
        user_items_names = user_items_names.iloc[sorted_indices]

        return user_items_names

# Maybe later change the get_recommendation to be able to receive a list
