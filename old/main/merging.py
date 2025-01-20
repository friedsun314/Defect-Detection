import numpy as np

def merge_similar_regions(image, regions, similarity_threshold=5.0):
    """
    Merges any two regions whose mean intensities differ by <= similarity_threshold.
    Repeats until no more merges can be done.
    
    :param image: Grayscale image (H x W).
    :param regions: List of sets, each containing (x,y) pixel coords.
    :param similarity_threshold: Merge if |mean1 - mean2| <= similarity_threshold.
    :return: A new list of merged region sets.
    """
    # Compute mean intensities
    region_means = []
    for reg in regions:
        vals = [image[y, x] for (x, y) in reg]
        region_means.append(np.mean(vals))

    merged = True
    while merged:
        merged = False
        skip = set()
        new_regions = []
        new_means = []

        for i in range(len(regions)):
            if i in skip:
                continue
            for j in range(i + 1, len(regions)):
                if j in skip:
                    continue
                diff = abs(region_means[i] - region_means[j])
                if diff <= similarity_threshold:
                    # Merge j into i
                    regions[i] = regions[i].union(regions[j])
                    # Recompute mean
                    vals = [image[y, x] for (x, y) in regions[i]]
                    region_means[i] = np.mean(vals)
                    skip.add(j)
                    merged = True
            
            # We'll finalize i after attempting merges
        # Rebuild region list
        for idx in range(len(regions)):
            if idx not in skip:
                new_regions.append(regions[idx])
                new_means.append(region_means[idx])

        regions = new_regions
        region_means = new_means

    return regions