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