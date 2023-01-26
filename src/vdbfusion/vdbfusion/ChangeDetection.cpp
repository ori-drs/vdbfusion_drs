
#include <openvdb/openvdb.h>
#include <openvdb/tools/Composite.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/ChangeBackground.h>

int main(){
//read vdb files

// Create a VDB file object.
openvdb::io::File file1("/home/joe/git/change_detection_drs/results/mickie_scans_100_1.vdb");
openvdb::io::File file2("/home/joe/git/change_detection_drs/results/mickie_scans_100_1.vdb");
// Open the file.  This reads the file header, but not any grids.
file1.open();
file2.open();
// Loop over all grids in the file and retrieve a shared pointer
// to the one named "LevelSetSphere".  (This can also be done
// more simply by calling file.readGrid("LevelSetSphere").)
openvdb::GridBase::Ptr Grid1;
for (openvdb::io::File::NameIterator nameIter = file1.beginName();
    nameIter != file1.endName(); ++nameIter)
  {
  Grid1 = file1.readGrid(nameIter.gridName());
  }

openvdb::GridBase::Ptr Grid2;
for (openvdb::io::File::NameIterator nameIter = file2.beginName();
    nameIter != file2.endName(); ++nameIter)
  {
  Grid2 = file2.readGrid(nameIter.gridName());
  }


// //
//  openvdb::FloatGrid::ConstPtr
//         copyOfGridA = Grid1->deepCopy(),
//         copyOfGridB = Grid2->deepCopy();
//     // At each voxel, compute a = max(a, b).
//     // openvdb::tools::compMax(*gridA, *gridB);
//     // // Restore the original grids.
//     // gridA = copyOfGridA->deepCopy();
//     // gridB = copyOfGridB->deepCopy();
//     // // At each voxel, compute a = min(a, b).
//     // openvdb::tools::compMin(*gridA, *gridB);
//     // // Restore the original grids.
//     // gridA = copyOfGridA->deepCopy();
//     // gridB = copyOfGridB->deepCopy();
//     // At each voxel, compute a = a + b.
//     // openvdb::tools::compSum(*gridA, *gridB);
//     openvdb::tools::csgDifference(*Grid1, *Grid2);



struct Local {
    static inline void diff(const float& a, const float& b, float& result) {
        result = a - b;
    }
};
openvdb::FloatGrid::Ptr resultGrid = openvdb::FloatGrid::create();
// Combine aGrid and bGrid and write the result into resultGrid.
// The input grids are not modified.
resultGrid->tree().combine2(Grid1->tree(), Grid2->tree(), Local::diff);


file1.close();
file2.close();
};