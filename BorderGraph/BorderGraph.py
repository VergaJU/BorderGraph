import numpy as np
import scanpy as sc
from anndata import AnnData
from joblib import Parallel, delayed
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from skimage import measure
from scipy.spatial import cKDTree
from itertools import combinations
import pickle
from typing import Dict, List, Tuple, Union
from typing import Optional
import os

class Utils():
    @staticmethod
    def get_labels(adata: AnnData, 
                    library_id: str, 
                    cell_key: str = 'Cell_ID',
                    library_key: str = 'library_id',
                    spatial_key: str = 'spatial',
                    image_key: str = 'images',
                    segmentation_key: str = 'segmentation',
                    **kwargs) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Retrieve the image and labels for a specific library ID from the given AnnData object.

        Parameters:
            adata (AnnData): The AnnData object containing the data.
            library_id (str): The ID of the library to retrieve the image and labels for.
            cell_key (str, optional): The key for the cell ID column in the AnnData object. Defaults to 'Cell_ID'.
            library_key (str, optional): The key for the library ID column in the AnnData object. Defaults to 'library_id'.
            spatial_key (str, optional): The key for the spatial data in the AnnData object. Defaults to 'spatial'.
            image_key (str, optional): The key for the images in the spatial data. Defaults to 'images'.
            segmentation_key (str, optional): The key for the segmentation data in the images. Defaults to 'segmentation'.

        Returns:
            Tuple[np.ndarray, Dict[str, int]]: A tuple containing the image and labels.
                - image (np.ndarray): The image data.
                - labels (Dict[str, int]): A dictionary mapping cell IDs to labels.
        """
        image = adata.uns[spatial_key][library_id][image_key][segmentation_key]
        labels = adata[adata.obs[library_key] == library_id].obs[cell_key].to_dict()
        return image, labels

    @staticmethod
    def create_csr_matrix(adata: AnnData, results: List[List[Union[int, float]]], adj: bool = False, **kwargs) -> csr_matrix:
        """
        Create a CSR matrix from the results.

        Parameters:
            adata (AnnData): The AnnData object.
            results (List[List[Union[int, float]]]): The results containing the indices and distances.
            adj (bool, optional): Whether to create an adjacency matrix. Defaults to False.

        Returns:
            csr_matrix: The CSR matrix.
        """
        rows = [obs[0] for obs in results] + [obs[1] for obs in results]
        cols = [obs[1] for obs in results] + [obs[0] for obs in results]
        if adj:
            data = [1] * len(rows)
        else:
            data = [obs[2] for obs in results] + [obs[2] for obs in results]
        adj = csr_matrix((data, (rows, cols)), shape=(adata.shape[0], adata.shape[0]))
        return adj

    @classmethod
    def update_adata(cls, adata: AnnData, key_added: str, results: List[List[Union[int, float]]], cutoff: float, **kwargs) -> AnnData:
        """
        Update the AnnData object with the computed results.

        Parameters:
            adata (AnnData): The AnnData object.
            key_added (str): The key to add to the AnnData object.
            results (List[List[Union[int, float]]]): The computed results.
            cutoff (float): The cutoff value.

        Returns:
            AnnData: The updated AnnData object.
        """
        params_key = key_added + '_neighbors'
        connectivities_key = key_added + '_connectivities'
        distance_key = key_added + '_distances'
        params_dict = {
            'connectivities_key': connectivities_key,
            'distances_key': distance_key,
            'params': {
                'n_neighbors': None,
                'coord_type': 'generic',
                'radius': cutoff,
                'transform': None
            }
        }
        adata.uns[params_key] = params_dict
        adata.obsp[connectivities_key] = cls.create_csr_matrix(adata, results, adj=True)
        adata.obsp[distance_key] = cls.create_csr_matrix(adata, results, adj=False)
        return adata

class Contours():
    @staticmethod
    def find_contours(image: np.ndarray, cell_id: str, label: int, offset: Tuple[int, int] = (0, 0), cutoff: float = 0.9, **kwargs) -> Dict[str, np.ndarray]:
        """
        Find contours for a specific label in the image.

        Parameters:
            image (np.ndarray): The image data.
            cell_id (str): The cell ID.
            label (int): The label to find contours for.
            offset (Tuple[int, int]): The offset to add to the contour coordinates. Defaults to (0, 0).
            cutoff (float, optional): The cutoff value. Defaults to 0.9.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the cell ID as key and the contours as value.
        """
        contour = measure.find_contours(image == label, cutoff)
        if len(contour) == 0:
            contour=np.array([[0,0]])
        else:
            contour = np.vstack(contour)
            contour[:, 0] += offset[0]
            contour[:, 1] += offset[1]
        return {cell_id: contour}

    @classmethod
    def parallel_find_contours(cls, image: np.ndarray, labels: Dict[str, int], cutoff: float = 0.9, n_jobs: int = -1, **kwargs) -> List[Dict[str, np.ndarray]]:
        """
        Find contours for multiple labels in parallel.

        Parameters:
            image (np.ndarray): The image data.
            labels (Dict[str, int]): A dictionary mapping cell IDs to labels.
            cutoff (float, optional): The cutoff value. Defaults to 0.9.
            n_jobs (int, optional): The number of parallel jobs. Defaults to -1.

        Returns:
            List[Dict[str, np.ndarray]]: A list of dictionaries containing the cell IDs as keys and the contours as values.
        """
        regions = measure.regionprops(image)
        label_to_region = {region.label: region for region in regions}
        
        tasks = []
        for cell_id, label in labels.items():
            try:
                label_int = int(label)
            except ValueError:
                continue # Skip if label cannot be converted to int
                
            if label_int in label_to_region:
                region = label_to_region[label_int]
                min_row, min_col, max_row, max_col = region.bbox
                # Add padding
                pad = 1
                min_row = max(0, min_row - pad)
                min_col = max(0, min_col - pad)
                max_row = min(image.shape[0], max_row + pad)
                max_col = min(image.shape[1], max_col + pad)
                
                cropped_image = image[min_row:max_row, min_col:max_col]
                offset = (min_row, min_col)
                tasks.append((cropped_image, cell_id, label_int, offset))
                
        results = Parallel(n_jobs=n_jobs)(delayed(cls.find_contours)(img, cid, lbl, off, cutoff) for img, cid, lbl, off in tasks)
        return results

    @classmethod
    def contours_per_image(cls, 
                           adata: AnnData, 
                           library_key: str='library_id',
                           cutoff: float = 0.9,
                           n_jobs: int = -1, 
                           copy: bool = True, 
                           contours_key: str = 'contours',
                           save: bool = True,
                           filename: str = 'contours.pkl',
                           **kwargs) -> Union[AnnData, Dict[str, Dict[str, np.ndarray]]]:
        """
        Compute contours for each image in the AnnData object.

        Parameters:
            adata (AnnData): The AnnData object.
            library_key (str): The key for the library ID column in the AnnData object.
            cutoff (float, optional): The cutoff value. Defaults to 0.9.
            n_jobs (int, optional): The number of parallel jobs. Defaults to -1.
            copy (bool, optional): Whether to create a copy of the AnnData object. Defaults to True.
            contours_key (str, optional): The key to store the computed contours in the AnnData object. Defaults to 'contours'.
            save (bool, optional): Whether to save the computed contours to a file. Defaults to True.
            filename (str, optional): The filename to save the contours. Defaults to 'countours.pkl'.

        Returns:
            Union[AnnData, Dict[str, Dict[str, np.ndarray]]]: If copy is True, returns the updated AnnData object.
                Otherwise, returns a dictionary containing the computed contours.
        """
        libraries = adata.obs[library_key].unique()
        contours = {}
        for library in libraries:
            image, labels = Utils.get_labels(adata, library)
            contours[library] = {}
            contours_list = cls.parallel_find_contours(image, labels, cutoff, n_jobs)
            for d in contours_list:
                contours[library].update(d)
        if save:
            with open(filename, 'wb') as handle:
                pickle.dump(contours, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if copy:
            adata.uns[contours_key] = contours
            return adata
        return contours


class Distances():
    @staticmethod
    def get_pairs(contours: Dict[str, np.ndarray], cutoff: float, **kwargs) -> List[Tuple[str, str]]:
        """
        Get pairs of labels from the contours that are potential candidates for being within cutoff distance.
        Uses a KDTree to filter pairs based on centroids and radii.

        Parameters:
            contours (Dict[str, np.ndarray]): A dictionary containing the labels as keys and the contours as values.
            cutoff (float): The cutoff distance.

        Returns:
            List[Tuple[str, str]]: A list of tuples representing the pairs of labels.
        """
        ids = list(contours.keys())
        centroids = []
        radii = []
        valid_indices = []
        
        for idx, i in enumerate(ids):
            c = contours[i]
            if len(c) <= 1: # Skip empty or single point placeholders if they are invalid
                continue
                
            centroid = np.mean(c, axis=0)
            # Max distance from centroid
            dists = np.linalg.norm(c - centroid, axis=1)
            radius = np.max(dists)
            
            centroids.append(centroid)
            radii.append(radius)
            valid_indices.append(idx)
            
        centroids = np.array(centroids)
        radii = np.array(radii)
        
        if len(centroids) == 0:
            return []
            
        tree = cKDTree(centroids)
        
        # Use a safe upper bound for the query radius: 2 * max_radius + cutoff
        max_r = np.max(radii)
        search_radius = 2 * max_r + cutoff
        
        candidate_indices = tree.query_pairs(search_radius)
        
        pairs = []
        for i, j in candidate_indices:
            dist = np.linalg.norm(centroids[i] - centroids[j])
            if dist < radii[i] + radii[j] + cutoff:
                # Map back to original IDs
                id_i = ids[valid_indices[i]]
                id_j = ids[valid_indices[j]]
                pairs.append((id_i, id_j))
                
        return pairs
        
    @staticmethod
    def compute_min_distance(pair: Tuple[str, str],
                             contours: Dict[str, np.ndarray],
                             cutoff: float, **kwargs) -> Optional[List[Union[str, float]]]:
        """
        Compute the minimum distance between two pairs of contours.

        Parameters:
            pair (Tuple[str, str]): The pair of labels.
            contours (Dict[str, np.ndarray]): A dictionary containing the labels as keys and the contours as values.
            cutoff (float): The cutoff value.

        Returns:
            Optional[List[Union[str, float]]]: A list containing the pair of labels and the minimum distance,
                or None if the minimum distance is greater than the cutoff.
        """
        dist_matrix = distance.cdist(contours[pair[0]], contours[pair[1]])
        min_dist = np.min(dist_matrix)
        if min_dist > cutoff:
            return None
        return [pair[0], pair[1], min_dist]

    @classmethod
    def process_chunk(cls,
                      chunk: List[Tuple[str, str]],
                      contours: Dict[str, np.ndarray],
                      cutoff: float, **kwargs) -> List[Optional[List[Union[str, float]]]]:
        """
        Process a chunk of pairs of contours and compute the minimum distance.

        Parameters:
            cls (class): The class object.
            chunk (List[Tuple[str, str]]): The chunk of pairs of labels.
            contours (Dict[str, np.ndarray]): A dictionary containing the labels as keys and the contours as values.
            cutoff (float): The cutoff value.

        Returns:
            List[Optional[List[Union[str, float]]]]: A list containing the pair of labels and the minimum distance,
                or None if the minimum distance is greater than the cutoff.
        """
        return [cls.compute_min_distance(pair, contours, cutoff) for pair in chunk]
                
    @classmethod
    def compute_image_pairs(cls,
                            adata: AnnData,
                            library_id: str,
                            cutoff: float,
                            contours_key: str = 'contours',
                            n_jobs: int = -1,
                            **kwargs) -> List[List[Union[str, float]]]:
        """
        Compute the pairs of labels and their minimum distances for a specific library ID.

        Parameters:
            adata (AnnData): The AnnData object.
            library_id (str): The library ID.
            contours_key (str, optional): The key for the computed contours in the AnnData object. Defaults to 'contours'.
            cutoff (float): The cutoff value.
            n_jobs (int, optional): The number of parallel jobs. Defaults to -1.

        Returns:
            List[List[Union[str, float]]]: A list containing the pairs of labels and their minimum distances.
        """
        contours = adata.uns[contours_key][library_id]
        pairs = cls.get_pairs(contours, cutoff)

        if not pairs:
            return []

        # Split pairs into chunks for parallel processing
        num_chunks = n_jobs if n_jobs > 0 else (os.cpu_count() or 1)
        # Handle case where pairs might be fewer than num_chunks
        num_chunks = min(num_chunks, len(pairs))
        chunk_size = int(np.ceil(len(pairs) / num_chunks))
        chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]

        # Parallel processing of chunks
        results = Parallel(n_jobs=n_jobs)(delayed(cls.process_chunk)(chunk, contours, cutoff) for chunk in chunks)

        # Flatten the list of results
        results = [item for sublist in results for item in sublist if item is not None]

        return results


    @classmethod
    def compute_all_images(cls,
                           adata: AnnData,
                           cutoff: float = 30,
                           contours_key: str = 'contours',
                           n_jobs: int = -1,
                           **kwargs) -> List[List[Union[int, float]]]:
        """
        Compute the pairs of labels and their minimum distances for all images in the AnnData object.

        Parameters:
            adata (AnnData): The AnnData object.
            contours_key (str, optional): The key for the computed contours in the AnnData object. Defaults to 'contours'.
            cutoff (float, optional): The cutoff value. Defaults to 30.
            n_jobs (int, optional): The number of parallel jobs. Defaults to -1.

        Returns:
            List[List[Union[int, float]]]: A list containing the pairs of labels and their minimum distances.
        """
        libraries = list(adata.uns[contours_key].keys())
        results = []
        for library in libraries:
            results.extend(cls.compute_image_pairs(adata, library, cutoff, contours_key, n_jobs))
        map_dict = {n: i for i, n in enumerate(adata.obs_names)}
        results = [[map_dict[obs[0]], map_dict[obs[1]], obs[2]] for obs in results if obs[0] in map_dict and obs[1] in map_dict]
        return results


def border_graph(adata: AnnData,
                 library_key: str = 'library_id',
                 cutoff: float = 30,
                 key_added: str = 'spatial',
                 **kwargs) -> AnnData:
    """
    Compute the border graph for an AnnData object.

    Parameters:
        adata (AnnData): The AnnData object.
        library_key (str, optional): The key for the library ID column in the AnnData object. Defaults to 'library_id'.
        cutoff (float, optional): The cutoff value. Defaults to 30.
        key_added (str, optional): The key to store the computed border graph in the AnnData object. Defaults to 'spatial'.
        **kwargs: Additional keyword arguments.

    Returns:
        AnnData: The updated AnnData object with the computed border graph.
    """
    adata = Contours.contours_per_image(adata, library_key, **kwargs)
    results = Distances.compute_all_images(adata, cutoff, **kwargs)
    adata = Utils.update_adata(adata, key_added, results, cutoff, **kwargs)
    return adata
