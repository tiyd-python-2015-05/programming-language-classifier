import web_scraper
import nose

"""public static void KnuthShuffle<T>(T[] array){    System.Random random = new System.Random();    for (int i = 0; i < array.Length; i++)    {        int j = random.Next(i, array.Length); // Don't select from the entire array on subsequent loops        T temp = array[i]; array[i] = array[j]; array[j] = temp;    }}
[edit] Clojure
(defn shuffle [vect]  (reduce (fn [v i] (let [r (rand-int i)]                      (assoc v i (v r) r (v i)))          vect (range (dec (count vect)) 1 -1)))
This works by generating a sequence of end-indices from n-1 to 1, then reducing that sequence (starting with the original vector) through a function that, given a vector and end-index, performs a swap between the end-index and some random index less than the end-index.

[edit] COBOL
       IDENTIFICATION DIVISION.       PROGRAM-ID. knuth-shuffle.        DATA DIVISION.       LOCAL-STORAGE SECTION.       01  i                       PIC 9(8).       01  j                       PIC 9(8).        01  temp                    PIC 9(8).        LINKAGE SECTION.       78  Table-Len               VALUE 10.       01  ttable-area.           03  ttable              PIC 9(8) OCCURS Table-Len TIMES.        PROCEDURE DIVISION USING ttable-area.           MOVE FUNCTION RANDOM(FUNCTION CURRENT-DATE (11:6)) TO i            PERFORM VARYING i FROM Table-Len BY -1 UNTIL i = 0               COMPUTE j =                   FUNCTION MOD(FUNCTION RANDOM * 10000, Table-Len) + 1                MOVE ttable (i) TO temp               MOVE ttable (j) TO ttable (i)               MOVE temp TO ttable (j)           END-PERFORM            GOBACK           .
[edit] CMake
# shuffle(<output variable> [<value>...]) shuffles the values, and# stores the result in a list.function(shuffle var)  set(forever 1)   # Receive ARGV1, ARGV2, ..., ARGV${last} as an array of values.  math(EXPR last "${ARGC} - 1")   # Shuffle the array with Knuth shuffle (Fisher-Yates shuffle"""


#looks like capture between [edit] \n   and \n
#should just capture for all languages
#do it for 50 programs
