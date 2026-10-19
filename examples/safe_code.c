#include <stdlib.h>
#include <string.h>

int safe_string_copy(char *dest, size_t dest_size, const char *src)
{
    if (!dest || !src || dest_size == 0)
        return -1;

    strncpy(dest, src, dest_size - 1);
    dest[dest_size - 1] = '\0';
    return 0;
}
