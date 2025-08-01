/*
 * Argon2 source code package
 * 
 * Written by Daniel Dinu and Dmitry Khovratovich, 2015
 * 
 * This work is licensed under a Creative Commons CC0 1.0 License/Waiver.
 * 
 * You should have received a copy of the CC0 Public Domain Dedication along with
 * this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
 */


#ifndef __ARGON2_KAT_H__
#define __ARGON2_KAT_H__

#include <string>
#include "argon2/argon2.h"
#include "argon2/argon2-core.h"
/*
 * Initial KAT function that prints the inputs to the file
 * @param  blockhash  Array that contains pre-hashing digest
 * @param  context Holds inputs
 * @param  type Argon2 type
 * @pre blockhash must point to INPUT_INITIAL_HASH_LENGTH bytes
 * @pre context member pointers must point to allocated memory of size according to the length values
 */
void InitialKat(const uint8_t* blockhash, const Argon2_Context* context, Argon2_type type);

/*
 * Function that prints the output tag
 * @param  out  output array pointer
 * @param  outlen digest length
 * @pre out must point to @a outlen bytes
 **/
void PrintTag(const void* out, uint32_t outlen);

/*
 * Function that prints the internal state at given moment
 * @param  instance pointer to the current instance
 * @param  pass current pass number
 * @pre instance must have necessary memory allocated
 **/
void InternalKat(const Argon2_instance_t* instance, uint32_t pass);


/*Generate test vectors of Argon2 of type @type
 * 
 */
void GenerateTestVectors(const std::string &type);

#endif
