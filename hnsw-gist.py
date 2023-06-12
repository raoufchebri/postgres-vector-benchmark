import h5py
import psycopg2
import time
import numpy

# For best performance:
# - Use socket connection
# - Use prepared statements (todo)
# - Use binary format (todo)
# - Increase work_mem? (todo)
# - Increase shared_buffers? (todo)
# - Try different drivers (todo)
class HNSW:
    def __init__(self, maxelements, dims):
        self._maxelements = maxelements
        self._dims = dims

        self._query = "SELECT id FROM t ORDER BY vec <-> %%s LIMIT %%s"

        self._conn = psycopg2.connect('postgresql://raouf@ep-curly-king-722988.eu-west-1.aws.neon.build/neondb') # psycopg2.connect('port=55432 user=cloud_admin dbname=postgres')
        self._conn.autocommit = True
        self._cur = self._conn.cursor()
        self._cur.execute('CREATE EXTENSION IF NOT EXISTS hnsw')

    def fit(self, X):
        self._cur.execute('CREATE TABLE IF NOT EXISTS t(id integer, vec real[])')

        f = open('gist.csv','w+')
        for i, x in enumerate(X):
            f.write(str(i))
            f.write('\t')
            f.write('{' + ','.join(str(v) for v in x.tolist()) + '}')
            f.write('\n')
        f.seek(0)
        self._cur.copy_from(f, 't', columns=('id', 'vec'))

        self._cur.execute('CREATE INDEX ON t USING hnsw (vec) WITH (maxelements = %d, dims=%d, m=32, efconstruction=16, efsearch=64)' % (self._maxelements, self._dims))
        self._cur.execute('ANALYZE t')

    def set_query_arguments(self):
        self._cur.execute('SET effective_io_concurrency=10')
        self._cur.execute('SET enable_seqscan=off')

    def query(self, v, n):
        self._cur.execute(self._query, (v.tolist(), n))
        res = self._cur.fetchall()
        return [r[0] for r in res]

    def __str__(self):
        return 'Pgvector(lists=%d, probes=%d)' % (self._lists, self._probes)


def get_dataset(hdf5_fn):
    hdf5_f = h5py.File(hdf5_fn, 'r')
    dimension = int(hdf5_f.attrs['dimension']) if 'dimension' in hdf5_f.attrs else len(hdf5_f['train'][0])
    return hdf5_f, dimension

def main():
    knn = 10 #  Number of nearest neighbours for the algorithm to return.',
    dataset = 'gist-960-euclidean.hdf5'
    D, dimension = get_dataset(dataset)
    X_train = numpy.array(D['train'])
    X_test = numpy.array(D['test'])
    n_elements = X_train.shape[0]
    algo = HNSW(n_elements, dimension)
    # print('got a train set of size (%d * %d)' % (n_elements, dimension))
    print('got %d queries' % len(X_test))

    start_time = time.time()
    algo.fit(X_train)
    build_time = time.time()
    print('Built index in', build_time - start_time)

    algo.set_query_arguments()
    for x in X_test:
        algo.query(x, knn)

    query_time =  time.time() - build_time
    print('Run queries in', query_time)


main()

