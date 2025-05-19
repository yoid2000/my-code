import s2cell

# The maximum level supported within an S2 cell ID. Each level is represented by two bits in the
# final cell ID
_S2_MAX_LEVEL = 30

# The maximum value within the I and J bits of an S2 cell ID
_S2_MAX_SIZE = 1 << _S2_MAX_LEVEL

# The number of bits in a S2 cell ID used for specifying the base face
_S2_FACE_BITS = 3

# The number of bits in a S2 cell ID used for specifying the position along the Hilbert curve
_S2_POS_BITS = 2 * _S2_MAX_LEVEL + 1

class s2stuff:
    def __init__(self):
        pass

    def cellId2Lat(self, cell_id):
        cell_id = int(cell_id)
        cell_id = self.make_cell_id_valid(cell_id)
        (lat,lon) = s2cell.cell_id_to_lat_lon(int(cell_id))
        return lat

    def cellId2Lon(self, cell_id):
        cell_id = int(cell_id)
        cell_id = self.make_cell_id_valid(cell_id)
        (lat,lon) = s2cell.cell_id_to_lat_lon(int(cell_id))
        return lon

    def cellId2LatLon(self, cell_id):
        cell_id = int(cell_id)
        cell_id = self.make_cell_id_valid(cell_id)
        (lat,lon) = s2cell.cell_id_to_lat_lon(int(cell_id))
        return (lat,lon)

    def lat_lon_to_cell_id(self, lat, lon):
        cell_id = s2cell.lat_lon_to_cell_id(lat, lon)
        return cell_id

    def make_cell_id_valid(self, cell_id):
        # Because the anonymization can create an invalid cell_id, we modify
        # it to make it valid. So far as I know, this increases precision
        lowest_set_bit = cell_id & (~cell_id + 1)  # pylint: disable=invalid-unary-operand-type
        if not lowest_set_bit & 0x1555555555555555:
            new_low_bit = lowest_set_bit >> 1
            cell_id |= new_low_bit
        return cell_id

    def cell_id_is_valid(self, cell_id):
        # Check input
        if not isinstance(cell_id, int):
            print(f"Bad cell_id {cell_id}, not an int")
            return False

        # Check for zero ID
        # This avoids overflow warnings below when 1 gets added to max uint64
        if cell_id == 0:
            print(f"Bad cell_id {cell_id}, ==0")
            return False

        # Check face bits
        if (cell_id >> _S2_POS_BITS) > 5:
            print(f"Bad cell_id {cell_id}, Too many face bits?")
            return False

        # Check trailing 1 bit is in one of the even bit positions allowed for the 30 levels, using the
        # mask: 0b0001010101010101010101010101010101010101010101010101010101010101 = 0x1555555555555555
        lowest_set_bit = cell_id & (~cell_id + 1)  # pylint: disable=invalid-unary-operand-type
        print(f"{cell_id:016X} cell_id {cell_id}")
        print(f"{lowest_set_bit:016X} lowest_set_bit")
        print(f"{(~cell_id+1):016X} ~cell_id+1")
        print(f"{(lowest_set_bit & 0x1555555555555555):016X} lowest_set_bit & 0x1555555555555555")
        if not lowest_set_bit & 0x1555555555555555:
            print("Bad cell_id")
            return False

        return True  # Checks have passed, cell ID must be valid


import random
import statistics

if __name__ == "__main__":
    X = 7  # Number of significant digits
    N = 1000  # Number of samples

    s2 = s2stuff()
    lat_errors = []
    lon_errors = []

    for _ in range(N):
        # Generate random latitude and longitude
        lat = round(random.uniform(-90, 90), X)
        lon = round(random.uniform(-180, 180), X)

        # Convert to cell id and back
        cell_id = s2.lat_lon_to_cell_id(lat, lon)
        lat2, lon2 = s2.cellId2LatLon(cell_id)

        # Compute absolute error
        lat_error = abs(lat - lat2)
        lon_error = abs(lon - lon2)

        lat_errors.append(lat_error)
        lon_errors.append(lon_error)

    def print_stats(name, errors):
        print(f"{name} error statistics over {N} samples:")
        print(f"  Mean:    {statistics.mean(errors):.10f}")
        print(f"  Stddev:  {statistics.stdev(errors):.10f}")
        print(f"  Min:     {min(errors):.10f}")
        print(f"  Max:     {max(errors):.10f}")

    print_stats("Latitude", lat_errors)
    print_stats("Longitude", lon_errors)