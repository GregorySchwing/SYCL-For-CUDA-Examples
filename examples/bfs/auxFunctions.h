#ifndef AUX_H
#define AUX_H

#include "CSRGraphRep.h"
#include "config.h"
#include <cerrno>

#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>

CSRGraph createCSRGraphFromFile(const char *filename);
#endif